"""Resume the patched Pixal3D multi-view run from a saved high-resolution SHAPE sample: texture conditioning + texture sampling + decode
+ GLB only (about 8-10 min instead of 27). Runs under the Pixal3D venv. One shot, no retry.

    pixal3d_resume_tex.py --repo DIR --run-dir DIR --dino-dir DIR --views-dir DIR --shape-latent pxmem_1024_raw_shape_hr_samples.pt
                          --output out.glb [--seed 42] [--num-views 4]

Hooks: none of the raw-saver hooks (no decode_latent wrapper, no sampler wrapper); only Mesh.fill_holes is replaced by a logged skip
(cumesh init crashes on this ROCm build) which also saves the decoded raw mesh to <output>_raw_mesh.pt the moment it exists.
Set PIXAL3D_PROJ_ACC_DEVICE=cpu to keep the fused-feature accumulator in host memory (it takes the 4-6 GB accumulation off the GPU).
"""
import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--dino-dir", required=True)
    ap.add_argument("--views-dir", required=True)
    ap.add_argument("--shape-latent", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-views", type=int, default=None)
    a = ap.parse_args()

    os.environ.setdefault("ATTN_BACKEND", "sdpa")
    os.environ.setdefault("SPARSE_ATTN_BACKEND", "sdpa")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    sys.path.insert(0, a.repo)
    sys.path.insert(1, a.run_dir)
    os.chdir(a.repo)

    import pixal3d_entry as E
    E.install_lazy_models(1)
    import inference_mv as P
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = os.environ.get("PIXAL3D_ALLOC_CONF", "max_split_size_mb:2048")
    E.log_allocator_state("resume")
    import numpy as np
    import torch
    import o_voxel
    from pixal3d import memlog
    from pixal3d.modules.sparse import SparseTensor
    for c in P.IMAGE_COND_CONFIGS.values():
        c["model_name"] = a.dino_dir
    import pixal3d_glb_cpu
    pixal3d_glb_cpu.install()
    prefix = os.path.splitext(os.path.abspath(a.output))[0] + "_raw"
    E.install_fill_holes_only(prefix)

    E.log_allocator_state("resume2")      # again after all imports, right before the first GPU allocation
    torch.set_grad_enabled(False)
    d = torch.load(a.shape_latent)
    m = re.search(r"proj,\s*(\d+)\)", str(d.get("desc", "")))
    res = int(m.group(1)) if m else 1024
    print(f"[resume] shape latent {a.shape_latent}: {d['feats'].shape[0]:,} tokens, resolution {res}", flush=True)

    with memlog.stage("load"):
        pipeline = P.init_pipeline(a.run_dir, "pipeline_mv.json", low_vram=True)
    memlog.install(pipeline)
    with memlog.stage("views"):
        views = P.load_views(a.views_dir, a.num_views, rembg=P.make_rembg(pipeline))
    P.check_main_view(views)

    dev = pipeline.device
    std = torch.tensor(pipeline.shape_slat_normalization["std"])[None].to(dev)
    mean = torch.tensor(pipeline.shape_slat_normalization["mean"])[None].to(dev)
    shape_slat = SparseTensor(feats=d["feats"].to(dev) * std + mean, coords=d["coords"].to(dev))

    torch.manual_seed(a.seed)
    cond_tex = pipeline.get_proj_cond_shape(
        pipeline.image_cond_model_tex_1024, [views], shape_slat.coords,
        camera_angle_x=views["camera_angle_x"], distance=views["camera_distance"], mesh_scale=views["mesh_scale"],
        grid_resolution_override=res // 16)
    tex_slat = pipeline.sample_tex_slat(
        cond_tex, pipeline.models["tex_slat_flow_model_1024"], shape_slat,
        {"steps": 12, "guidance_strength": 1.0, "guidance_rescale": 0.0, "rescale_t": 3.0})
    del cond_tex
    torch.cuda.empty_cache()
    torch.save({"feats": tex_slat.feats.cpu(), "coords": tex_slat.coords.cpu(), "res": res}, prefix + "_tex_latent.pt")
    print(f"[raw] sampled texture latent saved -> {prefix}_tex_latent.pt", flush=True)
    mesh = pipeline.decode_latent(shape_slat, tex_slat, res)[0]
    print(f"[resume] decoded: {mesh.vertices.shape[0]:,} vertices, {mesh.faces.shape[0]:,} faces", flush=True)
    torch.save({"vertices": mesh.vertices.cpu(), "faces": mesh.faces.cpu(), "coords": mesh.coords.cpu(), "attrs": mesh.attrs.cpu(),
                "res": res}, prefix + "_decoded.pt")
    print(f"[raw] decoded mesh with voxel attributes saved -> {prefix}_decoded.pt", flush=True)

    with memlog.stage("to_glb"):
        glb = o_voxel.postprocess.to_glb(
            vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs, coords=mesh.coords,
            attr_layout=pipeline.pbr_attr_layout, grid_size=res, aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target=1000000, texture_size=4096, remesh=True, remesh_band=1, remesh_project=0, use_tqdm=True)
    rot = np.array([[-1, 0, 0, 0], [0, 0, -1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=np.float64)
    glb.apply_transform(rot)
    glb.export(a.output, extension_webp=True)
    print(f"[Done] GLB saved to: {a.output}", flush=True)


if __name__ == "__main__":
    main()
