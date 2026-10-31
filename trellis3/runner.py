"""
One place that owns a loaded pipeline and runs jobs against it.

generate.py builds a Runner, runs one job and exits. serve.py builds one and
keeps it alive. Both go through `Runner.generate()` so the two paths cannot
drift.

Environment setup (MPS fallback, attention backends, sys.path for TRELLIS.2
and stubs) happens in `bootstrap()` and must run before torch is imported;
generate.py and serve.py call it first thing.
"""

import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

WATCHDOG_SIGNATURES = ("non-zero size", "BVH needs at least 8 triangles")

WATCHDOG_HELP = (
    "\nERROR: The decoder produced an empty mesh.\n"
    "On Apple Silicon this is almost always the macOS GPU watchdog\n"
    "killing a long-running Metal kernel in the SLat decoder. The Metal\n"
    "error prints to stderr above (look for\n"
    "'kIOGPUCommandBufferCallbackErrorImpactingInteractivity') but does\n"
    "not raise a Python exception, so execution continues with empty\n"
    "tensors and crashes downstream.\n"
    "\n"
    "Workarounds, cheapest first:\n"
    "  1. Run headless — close the lid / unplug external displays and\n"
    "     re-run over SSH. The watchdog tightens with WindowServer load.\n"
    "  2. MTL_CAPTURE_ENABLED=1 python generate.py ...   (extends the\n"
    "     watchdog timeout as a side effect of Metal-debugger mode)\n"
    "  3. SPARSE_CONV_BACKEND=none python generate.py ... (slower path,\n"
    "     may not help if a single dispatch is the offender)\n"
)


class WatchdogError(RuntimeError):
    pass


def bootstrap():
    """Set backend env vars and sys.path. Safe to call more than once."""
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    os.environ.setdefault("ATTN_BACKEND", "sdpa")
    os.environ.setdefault("SPARSE_ATTN_BACKEND", "sdpa")
    try:
        import flex_gemm  # noqa: F401  (probe only: metallib may fail to load)
        os.environ.setdefault("SPARSE_CONV_BACKEND", "flex_gemm")
    except (ImportError, RuntimeError):
        os.environ.setdefault("SPARSE_CONV_BACKEND", "none")
    t2 = os.path.join(ROOT, "TRELLIS.2")
    stubs = os.path.join(ROOT, "stubs")
    if t2 not in sys.path:
        sys.path.insert(0, t2)
    if stubs not in sys.path:
        sys.path.append(stubs)
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)


def pick_device():
    import torch
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class Runner:
    def __init__(self, pipeline_types, resident=False, stage_hooks=True,
                 path="microsoft/TRELLIS.2-4B", decode_only=False, log=print):
        from trellis3 import memory

        self.log = log
        if isinstance(pipeline_types, str):
            pipeline_types = [pipeline_types]
        names = memory.DECODE_ONLY if decode_only else memory.models_for(pipeline_types)
        what = "decode only" if decode_only else ", ".join(pipeline_types)
        log(f"\nLoading pipeline ({what}): {len(names)} checkpoints, ~{memory.approx_gb(names):.1f} GB")
        t0 = time.time()
        self.pipeline = memory.load_pipeline(pipeline_types, path=path, resident=resident,
                                             decode_only=decode_only)
        log(f"Loaded in {time.time() - t0:.0f}s")
        self.device = pick_device()
        self.pipeline.to(self.device)
        log(f"Device: {self.device}  low_vram={self.pipeline.low_vram}")
        if stage_hooks:
            memory.install_stage_hooks(self.pipeline)
        self._memory = memory

    def sample(self, image, pipeline_type="512", seed=42, steps=None, max_num_tokens=49152,
               state=None, on_stage=None, release=False, min_free_gb=0.0):
        """Staged, resumable sampling (trellis3.stages). Returns the state dict."""
        from trellis3.stages import sample_stages
        self._memory.ensure_models(self.pipeline, pipeline_type)
        return sample_stages(self.pipeline, image, pipeline_type, seed=seed, steps=steps,
                             max_num_tokens=max_num_tokens, state=state, on_stage=on_stage,
                             release=release, min_free_gb=min_free_gb, log=self.log)

    def decode(self, state):
        """Decode a completed state (from sample() or a latents file)."""
        from trellis3.stages import decode_state
        try:
            mesh = decode_state(self.pipeline, state, log=self.log)
        except (IndexError, AssertionError) as e:
            if any(sig in str(e) for sig in WATCHDOG_SIGNATURES):
                raise WatchdogError(WATCHDOG_HELP) from e
            raise
        if mesh.vertices.shape[0] == 0 or mesh.faces.shape[0] == 0:
            raise WatchdogError(WATCHDOG_HELP)
        return mesh

    def generate(self, image, seed=42, pipeline_type="512", steps=None,
                 max_num_tokens=49152):
        """Run the pipeline on a PIL image. Returns the MeshWithVoxel."""
        self._memory.ensure_models(self.pipeline, pipeline_type)
        overrides = {"steps": steps} if steps else {}
        try:
            outputs = self.pipeline.run(
                image,
                seed=seed,
                pipeline_type=pipeline_type,
                sparse_structure_sampler_params=overrides,
                shape_slat_sampler_params=overrides,
                tex_slat_sampler_params=overrides,
                max_num_tokens=max_num_tokens,
            )
        except (IndexError, AssertionError) as e:
            if any(sig in str(e) for sig in WATCHDOG_SIGNATURES):
                raise WatchdogError(WATCHDOG_HELP) from e
            raise
        mesh = outputs[0] if isinstance(outputs, list) else outputs
        if mesh.vertices.shape[0] == 0 or mesh.faces.shape[0] == 0:
            raise WatchdogError(WATCHDOG_HELP)
        return mesh
