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


def _metal_bake_available():
    """Mirror upstream's detection: Metal backend + rasterizer present."""
    try:
        import o_voxel.postprocess
    except (ImportError, AttributeError):
        return False
    backend = getattr(o_voxel.postprocess, "_BACKEND", None)
    has_dr = getattr(o_voxel.postprocess, "_HAS_DR", False)
    use_metal = backend == "metal" and has_dr
    if use_metal and not getattr(o_voxel.postprocess, "_HAS_FLEX_GEMM", False):
        _patch_grid_sample_fallback(o_voxel.postprocess)
    return use_metal


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
    return fast_simplification.simplify(verts_np, faces_np, ratio)


def _fix_winding(sv, sf, enabled, log):
    if not enabled:
        return sv, sf
    from trellis3.postbake import fix_winding
    t0 = time.time()
    sv, sf, flipped = fix_winding(sv, sf)
    log(f"  Winding pass: {flipped:,} faces flipped ({time.time() - t0:.1f}s)")
    return sv, sf


def export_mesh(mesh_out, output, texture_size=1024, no_texture=False,
                bake_faces=200000, metallic_factor=0.0, gamma=None, fix_winding=True,
                uv_cone_deg=None, uv_smooth=None, log=print):
    """Write <output>.glb (+ textures) and <output>.obj. Returns dict of paths.

    metallic_factor: pins glTF metallicFactor on the Metal path (upstream 1.0;
        model metallic drifts high on MPS, so default 0.0). None = leave as is.
    gamma: optional 1/gamma curve on base colour for the Metal path, which
        writes linear values (the KDTree path already applies 2.2).
    fix_winding: consistent outward winding on the simplified mesh before bake.
    uv_cone_deg / uv_smooth: chart clustering knobs passed to to_glb; larger
        values mean fewer, bigger UV charts (upstream 90 / 1).
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
        use_metal = _metal_bake_available()
        if use_metal:
            try:
                log(f"\nBaking PBR textures via Metal ({texture_size}x{texture_size})...")
                import o_voxel
                from trellis3.postbake import apply_gamma, set_metallic_factor
                target_faces = min(bake_faces, len(faces))
                sv, sf = _simplify(verts, faces, target_faces, log)
                sv, sf = _fix_winding(sv, sf, fix_winding, log)
                simp_verts_t = torch.from_numpy(np.ascontiguousarray(sv)).float()
                simp_faces_t = torch.from_numpy(np.ascontiguousarray(sf).astype("int32"))
                uv_kwargs = {}
                if uv_cone_deg is not None:
                    uv_kwargs["mesh_cluster_threshold_cone_half_angle_rad"] = np.radians(uv_cone_deg)
                if uv_smooth is not None:
                    uv_kwargs["mesh_cluster_smooth_strength"] = uv_smooth
                glb = o_voxel.postprocess.to_glb(
                    vertices=simp_verts_t,
                    faces=simp_faces_t,
                    attr_volume=mesh_out.attrs.cpu(),
                    coords=mesh_out.coords.cpu(),
                    attr_layout=mesh_out.layout,
                    voxel_size=mesh_out.voxel_size,
                    aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                    decimation_target=target_faces,
                    texture_size=texture_size,
                    verbose=True,
                    **uv_kwargs,
                )
                if set_metallic_factor(glb, metallic_factor):
                    log(f"  metallicFactor pinned to {metallic_factor}")
                if apply_gamma(glb, gamma):
                    log(f"  gamma {gamma} applied to base colour")
                glb.export(glb_path)
                log(f"  Saved: {glb_path}")
                paths["glb"] = glb_path
                paths["bake"] = "metal"
            except RuntimeError as e:
                log(f"\n  Metal bake failed: {e}")
                log("  Falling back to KDTree texture baker...")
                use_metal = False

        if not use_metal:
            log(f"\nBaking PBR textures via KDTree ({texture_size}x{texture_size})...")
            from backends.texture_baker import uv_unwrap, bake_texture, export_glb_with_texture

            target_faces = min(bake_faces, len(faces))
            bv, bf = _simplify(verts, faces, target_faces, log)
            bv, bf = _fix_winding(bv, bf, fix_winding, log)
            log("  UV unwrapping with xatlas...")
            new_verts, new_faces, uvs, _ = uv_unwrap(bv, bf)
            log(f"  UV unwrap: {len(verts):,} -> {len(new_verts):,} vertices")
            base_color_img, mr_img, _ = bake_texture(
                new_verts, new_faces, uvs,
                mesh_out.coords.cpu().float().numpy(),
                mesh_out.attrs.cpu().float().numpy(),
                mesh_out.origin.cpu().float().numpy(),
                mesh_out.voxel_size,
                texture_size=texture_size,
            )
            tex_path = f"{output}_basecolor.png"
            PILImage.fromarray(base_color_img).save(tex_path)
            export_glb_with_texture(new_verts, new_faces, uvs, base_color_img, mr_img, glb_path)
            log(f"  Saved: {glb_path}")
            paths["glb"] = glb_path
            paths["basecolor"] = tex_path
            paths["bake"] = "kdtree"
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
