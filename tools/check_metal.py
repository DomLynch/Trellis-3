#!/usr/bin/env python3
"""
Report which acceleration backends this trellis-mac/Trellis-3 install will
actually use. Run inside the project venv:

    python tools/check_metal.py

Why: without `mtlgemm` the texture baker silently falls back to
torch.nn.functional.grid_sample, which leaves ring/grid artifacts on curved
surfaces (see upstream README). Without `mtldiffrast` it falls back to the
pure-Python KDTree baker. Neither fallback raises; you only notice in the
output. This script makes the state explicit before you spend five minutes
on a run.
"""

import importlib
import os
import platform
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "TRELLIS.2"))
sys.path.append(os.path.join(ROOT, "stubs"))


def probe(name):
    try:
        mod = importlib.import_module(name)
        return True, getattr(mod, "__file__", "") or ""
    except Exception as e:  # ImportError or metallib RuntimeError
        return False, f"{type(e).__name__}: {e}"


def main():
    print(f"Platform: {platform.platform()}  Python {platform.python_version()}")
    ok_torch, info = probe("torch")
    if ok_torch:
        import torch
        mps = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
        print(f"torch {torch.__version__}  MPS available: {mps}")
    else:
        print(f"torch: MISSING ({info})")

    checks = [
        ("mtlgemm", "flex_gemm", "sparse conv + clean grid_sample in the baker (no ring artifacts)"),
        ("mtldiffrast", "mtldiffrast", "Metal rasterizer for texture baking"),
        ("mtlbvh", "mtlbvh", "Metal BVH used to map texels back to the full-res mesh"),
        ("mtlmesh", "mtlmesh", "Metal mesh ops"),
        ("o_voxel.postprocess", "o_voxel.postprocess", "to_glb — Metal bake entry point"),
    ]
    problems = 0
    for label, module, why in checks:
        ok, info = probe(module)
        status = "OK " if ok else "MISSING"
        print(f"  [{status}] {label:<22} {why}")
        if not ok:
            problems += 1
            print(f"           -> {info}")

    ok, _ = probe("o_voxel.postprocess")
    if ok:
        import o_voxel.postprocess as pp
        backend = getattr(pp, "_BACKEND", None)
        has_dr = getattr(pp, "_HAS_DR", False)
        has_fg = getattr(pp, "_HAS_FLEX_GEMM", False)
        print(f"\no_voxel.postprocess: backend={backend} rasterizer={'yes' if has_dr else 'no'} "
              f"flex_gemm_grid_sample={'yes' if has_fg else 'no (ring artifacts likely)'}")
        if backend != "metal" or not has_dr:
            problems += 1

    conv = os.environ.get("SPARSE_CONV_BACKEND", "(unset: generate.py picks flex_gemm if importable)")
    print(f"SPARSE_CONV_BACKEND={conv}")

    print()
    if problems == 0:
        print("Full Metal stack active. Expect 'Baking PBR textures via Metal' in generate.py logs.")
    else:
        print(f"{problems} component(s) missing. Fix with:")
        print("    xcodebuild -downloadComponent MetalToolchain")
        print("    rm -rf deps && bash setup.sh")
    return 0 if problems == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
