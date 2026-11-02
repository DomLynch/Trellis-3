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


class LazyModels(dict):
    """Drop-in for Pipeline.models that loads a checkpoint on first use and keeps only `max_big` of the big
    flow models in RAM (LRU). Pixal3D-MV's four flow models are ~22 GB; eager loading does not fit a 32 GB
    machine next to the encoders. A model used again after eviction is simply reloaded from disk."""

    def __init__(self, base, spec, max_big=1, only=None):
        super().__init__()
        self.base, self.max_big = base, max_big
        self.spec = {k: v for k, v in spec.items() if only is None or k in only}
        self.lru = []

    def __contains__(self, k):
        return k in self.spec

    def __getitem__(self, k):
        if dict.__contains__(self, k):
            if k in self.lru:
                self.lru.remove(k)
                self.lru.append(k)
            return dict.__getitem__(self, k)
        import gc
        from pixal3d import models
        if "flow" in k:                      # make room first: the old model is dropped before the new one is read
            while len(self.lru) >= self.max_big:
                old = self.lru.pop(0)
                dict.__delitem__(self, old)
                print(f"[LazyModels] evicted {old}", flush=True)
            gc.collect()
        path = f"{self.base}/{self.spec[k]}"
        try:
            m = models.from_pretrained(path)
        except Exception:
            m = models.from_pretrained(self.spec[k])
        m.eval()
        if "flow" in k:
            self.lru.append(k)
        dict.__setitem__(self, k, m)
        print(f"[LazyModels] loaded {k}", flush=True)
        return m


def install_lazy_models(max_big=1):
    import json
    from pixal3d.pipelines import base

    def from_pretrained(cls, path, config_file="pipeline.json"):
        with open(f"{path}/{config_file}") as f:
            args = json.load(f)["args"]
        only = getattr(cls, "model_names_to_load", None)
        pipe = cls(LazyModels(path, args["models"], max_big, only))
        pipe._pretrained_args = args
        return pipe

    base.Pipeline.from_pretrained = classmethod(from_pretrained)


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
    ap.add_argument("--max-tokens", type=int, default=49152, help="cascade token budget (resolution drops when exceeded)")
    ap.add_argument("--eager", action="store_true", help="load every checkpoint up front (needs ~30 GB of RAM for the MV set)")
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

    if not a.eager:
        install_lazy_models(1)
    # inference*.py force expandable_segments, whose HIP mapping fails with 20+ GB free on this ROCm build
    # (OOM at a 6.75 GiB NAF allocation). Use the classic allocator and release cache between views.
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = os.environ.get("PIXAL3D_ALLOC_CONF", "max_split_size_mb:2048")
    if a.views_dir:
        import inference_mv as P
    else:
        import inference as P
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = os.environ.get("PIXAL3D_ALLOC_CONF", "max_split_size_mb:2048")  # inference*.py set it again at import
    try:
        import torch
        from pixal3d.trainers.flow_matching.mixins import image_conditioned_proj as ICP
        orig = ICP.DinoV3ProjMultiViewFeatureExtractor._project_single_view

        def per_view(self, *args, **kw):
            torch.cuda.empty_cache()
            out = orig(self, *args, **kw)
            torch.cuda.empty_cache()
            return out
        ICP.DinoV3ProjMultiViewFeatureExtractor._project_single_view = per_view
    except Exception as e:  # noqa: BLE001
        print(f"[entry] per-view cache release not installed: {e}", flush=True)
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
                        resolution=a.resolution, max_num_tokens=a.max_tokens)
    else:
        P.run_inference(image_path=a.image, output_path=a.output, seed=a.seed, manual_fov=-1.0,
                        model_path=a.run_dir, low_vram=a.low_vram, resolution=a.resolution, max_num_tokens=a.max_tokens)


if __name__ == "__main__":
    main()
