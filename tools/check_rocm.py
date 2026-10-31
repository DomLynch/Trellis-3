#!/usr/bin/env python3
"""
Report what Trellis-3 will use on an AMD (ROCm) or NVIDIA machine.

    python tools/check_rocm.py

Checks: torch build (HIP/CUDA), the GPU torch sees, bf16 matmul, which
scaled_dot_product_attention kernels are enabled (flash / mem-efficient /
math), and that the pure-PyTorch sparse-conv and CPU bake paths import.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "TRELLIS.2"))
sys.path.append(os.path.join(ROOT, "stubs"))
sys.path.insert(0, ROOT)


def main():
    import platform
    print(f"Platform: {platform.platform()}  Python {platform.python_version()}")
    try:
        import torch
    except ImportError:
        print("torch: MISSING"); return 1
    hip = getattr(torch.version, "hip", None)
    print(f"torch {torch.__version__}  build: {'ROCm ' + hip if hip else ('CUDA ' + str(torch.version.cuda) if torch.version.cuda else 'CPU-only')}")
    if not torch.cuda.is_available():
        print("torch.cuda.is_available() = False -> install the ROCm wheel for this GPU "
              "(see README 'AMD / ROCm'). Trellis-3 would run on CPU only.")
        return 1
    dev = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}  total {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    try:
        a = torch.randn(2048, 2048, device=dev, dtype=torch.bfloat16)
        torch.cuda.synchronize(); (a @ a).sum().item(); torch.cuda.synchronize()
        print("bf16 matmul: OK")
    except Exception as e:
        print(f"bf16 matmul: FAILED ({e})")
    try:
        from torch.backends.cuda import flash_sdp_enabled, math_sdp_enabled, mem_efficient_sdp_enabled
        print(f"SDPA kernels enabled: flash={flash_sdp_enabled()} mem_efficient={mem_efficient_sdp_enabled()} math={math_sdp_enabled()}")
        q = torch.randn(1, 12, 4096, 128, device=dev, dtype=torch.bfloat16)
        torch.cuda.reset_peak_memory_stats()
        torch.nn.functional.scaled_dot_product_attention(q, q, q); torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"SDPA 12x4096x128 peak {peak:.2f} GB -> {'fused (no score matrix)' if peak < 0.5 else 'math path (score matrix materialised; auto-tiling will engage at 1024)'}")
    except Exception as e:
        print(f"SDPA probe: FAILED ({e})")
    problems = 0
    for mod, why in (("trellis3.stages", "staged sampling"), ("backends.conv_none", "pure-PyTorch sparse conv"),
                     ("backends.texture_baker", "CPU texture bake"), ("backends.attention_mps", "tiled attention")):
        try:
            __import__(mod); print(f"  [OK ] {mod:<26} {why}")
        except Exception as e:
            problems += 1; print(f"  [MISSING] {mod:<22} {why}: {e}")
    try:
        import xatlas  # noqa: F401 — probe only
        del xatlas
        print("  [OK ] xatlas                     UV unwrap for the CPU bake")
    except ImportError:
        problems += 1; print("  [MISSING] xatlas                 pip install xatlas")
    print("\nSPARSE_CONV_BACKEND should be 'none' on ROCm (flex_gemm is CUDA/Metal only):",
          os.environ.get("SPARSE_CONV_BACKEND", "(unset)"))
    return 0 if problems == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
