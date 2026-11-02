"""
Mesh export shared by generate.py (one-shot) and serve.py (resident).

This is the texture-bake / GLB / OBJ tail of upstream generate.py lifted
into a function so a resident process can call it per job. Behaviour is
unchanged from upstream: Metal bake via o_voxel.postprocess.to_glb when the
Metal stack is present, KDTree baker otherwise, vertex-colour GLB as a last
resort, and an OBJ of the full-resolution geometry alongside.
"""

import os
import time

import numpy as np
import torch
from PIL import Image as PILImage


def gpu_bake_backend():
    """Which native `o_voxel.postprocess.to_glb` is usable, or None.

    "metal": Pedro Naugusto's Apple fork (sets _BACKEND/_HAS_DR; wants CPU tensors).
    "cuda":  Microsoft's o_voxel built for CUDA or ROCm (no _BACKEND attribute;
             needs nvdiffrast + cumesh and cuda tensors). The ROCm wheels from
             ComfyUI-Trellis2-AMD give exactly this on AMD.
    """
    try:
        import o_voxel.postprocess
    except (ImportError, AttributeError):
        return None
    pp = o_voxel.postprocess
    if hasattr(pp, "_BACKEND"):                       # Apple fork
        backend = getattr(pp, "_BACKEND", None)
        has_dr = getattr(pp, "_HAS_DR", False)
        if backend in ("metal", "cuda") and has_dr:
            if not getattr(pp, "_HAS_FLEX_GEMM", False):
                _patch_grid_sample_fallback(pp)
            return backend
        return None
    try:                                              # official o_voxel: needs its CUDA/HIP deps
        import nvdiffrast.torch  # noqa: F401
        import cumesh  # noqa: F401
    except Exception:
        return None
    return "cuda" if torch.cuda.is_available() else None


def gpu_bake_unavailable_reason():
    """Why gpu_bake_backend() is None, for the log line of the CPU fallback ('' when a GPU baker is usable)."""
    if gpu_bake_backend() is not None:
        return ""
    try:
        import o_voxel
        where = getattr(o_voxel, "__file__", None) or "?"
    except Exception as e:
        return f"o_voxel does not import ({e!r})"
    try:
        import o_voxel.postprocess  # noqa: F401
    except Exception as e:
        hint = " (o_voxel resolves to a STUB package: keep the repo's stubs/ dir off PYTHONPATH, bootstrap() appends it last)" if "stubs" in where.replace("\\", "/") else ""
        return f"o_voxel.postprocess does not import ({e!r}); o_voxel is {where}{hint}"
    for mod in ("nvdiffrast.torch", "cumesh"):
        try:
            __import__(mod)
        except Exception as e:
            return f"{mod} does not import ({e!r})"
    return "torch.cuda.is_available() is False" if not torch.cuda.is_available() else "backend probe returned None"


def _metal_bake_available():
    return gpu_bake_backend() is not None


def _patch_grid_sample_fallback(pp):
    """o_voxel's _grid_sample_3d fallback returns [B*C, M] but the bake
    consumes it as [M, C]. Patch it to transpose before the reshape (upstream
    generate.py does the same inline)."""
    import torch.nn.functional as F

    def _gs3d_fix(feats, coords, shape, grid, mode="trilinear"):
        B, C = shape[0], shape[1]
        D, H, W = shape[2], shape[3], shape[4]
        device = feats.device
        dense_vol = torch.zeros(B, C, D, H, W, dtype=feats.dtype, device=device)
        batch_idx = coords[:, 0].long()
        cx = coords[:, 1].long(); cy = coords[:, 2].long(); cz = coords[:, 3].long()
        dense_vol[batch_idx, :, cx, cy, cz] = feats
        grid_norm = torch.stack([
            grid[..., 2] / (W - 1) * 2 - 1,
            grid[..., 1] / (H - 1) * 2 - 1,
            grid[..., 0] / (D - 1) * 2 - 1,
        ], dim=-1).reshape(B, 1, 1, -1, 3)
        sampled = F.grid_sample(dense_vol, grid_norm, mode="bilinear",
                                align_corners=True, padding_mode="border")
        M = grid.shape[1]
        return sampled.reshape(B, C, M).permute(0, 2, 1).reshape(B * M, C)

    pp._grid_sample_3d = _gs3d_fix


def _simplify(verts_np, faces_np, target_faces, log):
    if len(faces_np) <= target_faces:
        return verts_np, faces_np
    try:
        import fast_simplification
    except ImportError:
        log("  Warning: fast_simplification not installed, using full mesh (slow)")
        return verts_np, faces_np
    ratio = 1.0 - (target_faces / len(faces_np))
    log(f"  Simplifying mesh: {len(faces_np):,} -> ~{target_faces:,} faces")
    t0 = time.time()
    sv, sf = fast_simplification.simplify(verts_np, faces_np, ratio)
    log(f"  Simplified: {len(sf):,} faces ({time.time() - t0:.0f}s)")   # stage line (issue-188 work): this step alone took ~18 min on a 70M-face hero mesh
    return sv, sf


def _fix_winding(sv, sf, enabled, log):
    if not enabled:
        return sv, sf
    from trellis3.postbake import fix_winding
    t0 = time.time()
    sv, sf, flipped = fix_winding(sv, sf)
    log(f"  Winding pass: {flipped:,} faces flipped ({time.time() - t0:.1f}s)")
    return sv, sf


def _hires_index(verts, faces, log):
    from trellis3.hires import NearestSurface
    t0 = time.time()
    ns = NearestSurface(verts, faces)
    log(f"  Full-res index: {len(faces):,} faces ({time.time() - t0:.1f}s)")
    return ns


def _bake_normals(obj, uvs, hires, texture_size, v_down, flip_v, in_glb_frame, log):
    """Bake + attach a normal map onto a textured Trimesh. Returns the PNG array."""
    from trellis3.hires import attach_normal_map, bake_normal_map, glb_to_decoder_frame
    V = np.asarray(obj.vertices)
    if in_glb_frame:
        V = glb_to_decoder_frame(V)
    uv = np.asarray(uvs, dtype=np.float32).copy()
    if flip_v:
        uv[:, 1] = 1.0 - uv[:, 1]
    log(f"  Baking normal map from full-res mesh ({texture_size}x{texture_size})...")
    img, _ = bake_normal_map(V, np.asarray(obj.faces), uv, hires, texture_size,
                             v_down=v_down, log=log)
    attach_normal_map(obj, img)
    return img


def export_mesh(mesh_out, output, texture_size=1024, no_texture=False,
                bake_faces=200000, metallic_factor=0.0, gamma=None, fix_winding=True,
                uv_cone_deg=None, uv_smooth=None, normal_map=False, normal_flip_v=False,
                snap_hires=True, reference_image=None, bake_sampling="trilinear", log=print):
    """Write <output>.glb (+ textures) and <output>.obj. Returns dict of paths.

    metallic_factor: pins glTF metallicFactor on the Metal path (upstream 1.0;
        model metallic drifts high on MPS, so default 0.0). None = leave as is.
    gamma: optional 1/gamma curve on base colour for the Metal path, which
        writes linear values (the KDTree path already applies 2.2). "auto"
        solves the gamma that matches the texture's mean colour to
        `reference_image` (the input photo); measured 1.6-1.7 on real heads.
    fix_winding: consistent outward winding on the simplified mesh before bake.
    uv_cone_deg / uv_smooth: chart clustering knobs passed to to_glb; larger
        values mean fewer, bigger UV charts (upstream 90 / 1).
    normal_map: bake a tangent-space normal map from the full-res decoder
        mesh (CPU nearest-surface) and attach it as normalTexture.
    normal_flip_v: flip V when rasterising the normal map (Metal-path UV
        convention is unverified; use this if the map looks vertically off).
    snap_hires: CPU path only — snap texel positions onto the full-res
        mesh before sampling voxel colour.
    bake_sampling: CPU path only — "trilinear" (the official bake's maths,
        default) or "knn" (upstream trellis-mac's inverse-distance average).
    """
    verts = mesh_out.vertices.cpu().numpy()
    faces = mesh_out.faces.cpu().numpy()
    if verts.shape[0] == 0 or faces.shape[0] == 0:
        raise RuntimeError("decoder produced an empty mesh")

    paths = {}
    glb_path = f"{output}.glb"
    has_voxels = getattr(mesh_out, "attrs", None) is not None
    t_bake = time.time()

    if has_voxels and not no_texture:
        gpu_backend = gpu_bake_backend()
        use_metal = gpu_backend is not None
        if use_metal:
            try:
                bake_dev = "cuda" if gpu_backend == "cuda" and torch.cuda.is_available() else "cpu"
                log(f"\nBaking PBR textures via {gpu_backend} ({texture_size}x{texture_size})...")
                import o_voxel
                from trellis3.postbake import apply_gamma, set_metallic_factor
                target_faces = min(bake_faces, len(faces))
                sv, sf = _simplify(verts, faces, target_faces, log)
                sv, sf = _fix_winding(sv, sf, fix_winding, log)
                simp_verts_t = torch.from_numpy(np.ascontiguousarray(sv)).float().to(bake_dev)
                simp_faces_t = torch.from_numpy(np.ascontiguousarray(sf).astype("int32")).to(bake_dev)
                uv_kwargs = {}
                if uv_cone_deg is not None:
                    uv_kwargs["mesh_cluster_threshold_cone_half_angle_rad"] = np.radians(uv_cone_deg)
                if uv_smooth is not None:
                    uv_kwargs["mesh_cluster_smooth_strength"] = uv_smooth
                from trellis3 import glb_lean
                to_glb, late = glb_lean.get_to_glb()   # attribute volume reaches the GPU only for the bake (issue 188)
                paths["glb_lean"] = late
                t_glb = time.time()
                log(f"  to_glb start ({len(sf):,} faces, texture {texture_size}); its verbose stage prints follow")
                glb = to_glb(
                    vertices=simp_verts_t,
                    faces=simp_faces_t,
                    attr_volume=mesh_out.attrs if late else mesh_out.attrs.to(bake_dev),
                    coords=mesh_out.coords if late else mesh_out.coords.to(bake_dev),
                    attr_layout=mesh_out.layout,
                    voxel_size=mesh_out.voxel_size,
                    aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                    decimation_target=target_faces,
                    texture_size=texture_size,
                    verbose=True,
                    **uv_kwargs,
                )
                log(f"  to_glb done ({time.time() - t_glb:.0f}s)")
                if set_metallic_factor(glb, metallic_factor):
                    log(f"  metallicFactor pinned to {metallic_factor}")
                if gamma == "auto":
                    from trellis3.postbake import auto_gamma
                    if reference_image is None:
                        log("  gamma auto: no reference image given, skipping")
                        gamma = None
                    else:
                        gamma, mt, mr = auto_gamma(glb, reference_image)
                        if gamma is not None:
                            log(f"  gamma auto: texture mean {np.round(mt, 3)} vs photo "
                                f"{np.round(mr, 3)} -> gamma {gamma:.2f}")
                if apply_gamma(glb, gamma):
                    log(f"  gamma {gamma} applied to base colour")
                paths["gamma"] = gamma
                if normal_map:
                    hires = _hires_index(verts, faces, log)
                    nimg = _bake_normals(glb, glb.visual.uv, hires, texture_size,
                                         v_down=False, flip_v=normal_flip_v,
                                         in_glb_frame=True, log=log)
                    PILImage.fromarray(nimg, "RGB").save(f"{output}_normal.png")
                    paths["normal"] = f"{output}_normal.png"
                glb.export(glb_path)
                log(f"  Saved: {glb_path}")
                paths["glb"] = glb_path
                paths["bake"] = gpu_backend
            except RuntimeError as e:
                log(f"\n  Metal bake failed: {e}")
                log("  Falling back to KDTree texture baker...")
                use_metal = False

        if not use_metal:
            why_cpu = "the GPU bake raised (see above)" if (has_voxels and gpu_bake_backend()) else (gpu_bake_unavailable_reason() or "not available")
            log(f"\nBaking PBR textures on CPU ({texture_size}x{texture_size}, {bake_sampling}); GPU baker not used: {why_cpu}")
            from backends.texture_baker import uv_unwrap, bake_texture, build_textured_mesh
            from trellis3.postbake import apply_gamma, auto_gamma, set_metallic_factor

            target_faces = min(bake_faces, len(faces))
            bv, bf = _simplify(verts, faces, target_faces, log)
            bv, bf = _fix_winding(bv, bf, fix_winding, log)
            log("  UV unwrapping with xatlas...")
            t_uv = time.time()
            new_verts, new_faces, uvs, _ = uv_unwrap(bv, bf)
            log(f"  UV unwrap: {len(verts):,} -> {len(new_verts):,} vertices ({time.time() - t_uv:.0f}s)")
            hires = _hires_index(verts, faces, log) if (snap_hires or normal_map) else None
            t_tex = time.time()
            log(f"  Baking base colour / metallic-roughness ({texture_size}x{texture_size}, {len(new_faces):,} faces)...")
            base_color_img, mr_img, _ = bake_texture(
                new_verts, new_faces, uvs,
                mesh_out.coords.cpu().float().numpy(),
                mesh_out.attrs.cpu().float().numpy(),
                mesh_out.origin.cpu().float().numpy(),
                mesh_out.voxel_size,
                texture_size=texture_size,
                snap=hires if snap_hires else None,
                sampling=bake_sampling,
                gamma=None,                       # applied below, same as the Metal path
            )
            log(f"  Texture bake done ({time.time() - t_tex:.0f}s)")
            tex_path = f"{output}_basecolor.png"
            normal_img = None
            if normal_map:
                from trellis3.hires import bake_normal_map
                log(f"  Baking normal map from full-res mesh ({texture_size}x{texture_size})...")
                normal_img, _ = bake_normal_map(new_verts, new_faces, uvs, hires, texture_size,
                                                v_down=True, log=log)
                PILImage.fromarray(normal_img, "RGB").save(f"{output}_normal.png")
                paths["normal"] = f"{output}_normal.png"
            glb = build_textured_mesh(new_verts, new_faces, uvs, base_color_img, mr_img, normal_img)
            if set_metallic_factor(glb, metallic_factor):
                log(f"  metallicFactor pinned to {metallic_factor}")
            g = gamma
            if g == "auto":
                g = None
                if reference_image is not None:
                    g, mt, mr = auto_gamma(glb, reference_image)
                    if g is not None:
                        log(f"  gamma auto: texture mean {np.round(mt, 3)} vs photo {np.round(mr, 3)} -> gamma {g:.2f}")
            if g is None and gamma != "auto":
                g = 2.2                            # upstream default when nothing else is asked
            if apply_gamma(glb, g):
                log(f"  gamma {g} applied to base colour")
            paths["gamma"] = g
            glb.visual.material.baseColorTexture.convert("RGB").save(tex_path)
            t_w = time.time()
            glb.export(glb_path)
            log(f"  GLB written ({time.time() - t_w:.0f}s)")
            log(f"  Saved: {glb_path}")
            paths["glb"] = glb_path
            paths["basecolor"] = tex_path
            paths["bake"] = f"cpu-{bake_sampling}"
        log(f"  Bake time: {time.time() - t_bake:.0f}s")
    else:
        log("\nExporting with vertex colors (use --texture-size for PBR textures)...")
        import trimesh
        trimesh.Trimesh(vertices=verts, faces=faces).export(glb_path)
        log(f"Saved: {glb_path}")
        paths["glb"] = glb_path
        paths["bake"] = "none"

    obj_path = f"{output}.obj"
    with open(obj_path, "w") as f:
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
    log(f"Saved: {obj_path}")
    paths["obj"] = obj_path
    paths["vertices"] = int(verts.shape[0])
    paths["faces"] = int(faces.shape[0])
    return paths


def output_stem(output, out_dir=None):
    """Resolve the output stem, creating out_dir if given."""
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        return os.path.join(out_dir, os.path.basename(output))
    return output
