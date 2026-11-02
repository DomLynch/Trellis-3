"""Run Pixal3D single-view or multi-view inference under the Pixal3D venv, without editing the upstream clone.
Started by trellis3/pixal3d.py (generate.py --preset pixal3d). Not for the Trellis-3 venv.

    pixal3d_entry.py --repo DIR --run-dir DIR --dino-dir DIR --output out.glb (--image a.png | --views-dir D --num-views N)
                     [--seed 42] [--resolution -1] [--low-vram] [--glb cpu|native]

In memory only:
  - DINOv3 is read from the local directory for every conditioning stage (no network, no token);
  - --run-dir holds the patched pipeline.json / pipeline_mv.json (local BiRefNet instead of the gated RMBG-2.0) and the ckpts junction;
  - --glb cpu (default) swaps o_voxel's to_glb for the numpy/CPU port in <run-dir>/pixal3d_glb_cpu.py, because cumesh's CuMesh.init
    crashes on this ROCm build (io.hip:134). It skips the dual-contouring remesh; label such results PX-CPUGLB.
natten (CUDA-only, needed by the NAF upsampler) comes from the pure-PyTorch shim the launcher puts on PYTHONPATH.
"""
import argparse
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--dino-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--image")
    ap.add_argument("--views-dir")
    ap.add_argument("--num-views", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resolution", type=int, default=-1)
    ap.add_argument("--low-vram", action="store_true")
    ap.add_argument("--glb", choices=["cpu", "native"], default="cpu")
    a = ap.parse_args()
    if bool(a.image) == bool(a.views_dir):
        sys.exit("give exactly one of --image or --views-dir")

    os.environ.setdefault("ATTN_BACKEND", "sdpa")
    os.environ.setdefault("SPARSE_ATTN_BACKEND", "sdpa")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    sys.path.insert(0, a.repo)
    sys.path.insert(1, a.run_dir)
    os.chdir(a.repo)

    if a.views_dir:
        import inference_mv as P
    else:
        import inference as P
    for c in P.IMAGE_COND_CONFIGS.values():
        c["model_name"] = a.dino_dir
    if a.glb == "cpu":
        import pixal3d_glb_cpu
        pixal3d_glb_cpu.install()
        print("GLB export: CPU port (PX-CPUGLB, no remesh)", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(a.output)), exist_ok=True)
    if a.views_dir:
        P.run_inference(views_dir=a.views_dir, output_path=a.output, num_views=a.num_views, seed=a.seed,
                        model_path=a.run_dir, config_file="pipeline_mv.json", low_vram=a.low_vram,
                        resolution=a.resolution)
    else:
        P.run_inference(image_path=a.image, output_path=a.output, seed=a.seed, manual_fov=-1.0,
                        model_path=a.run_dir, low_vram=a.low_vram, resolution=a.resolution)


if __name__ == "__main__":
    main()
