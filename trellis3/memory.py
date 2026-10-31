"""
Memory controls for TRELLIS.2 on unified-memory Macs.

What upstream already does
--------------------------
`Trellis2ImageTo3DPipeline` ships with `low_vram=True`: every model is kept on
CPU and copied to the device only for the stage that uses it. So "stage
offloading" exists. What it does NOT do, and what costs RAM on a 24GB Mac:

1. It loads all eight checkpoints (~15GB bf16/fp16) even though a single run
   uses five or six of them. The `512` pipeline never touches the two 1024
   flow models (5.2GB); `1024_cascade` never touches `tex_slat_flow_model_512`
   (2.6GB). `models_for()` + `load_pipeline()` only load what the run needs.

2. The MPS caching allocator keeps freed blocks. `install_stage_hooks()` calls
   `torch.mps.empty_cache()` after each stage so the next one starts clean.

3. `max_num_tokens` (default 49152) bounds the 1024/1536 cascade token count
   and is the single biggest lever on activation memory at high resolution.
   The pipeline already lowers the resolution automatically when the budget is
   exceeded; `generate.py --max-tokens` exposes it.

4. On unified memory the CPU<->MPS shuttle is pure copy cost. `resident=True`
   turns `low_vram` off so the selected models stay on MPS. Peak RAM is about
   the same (weights live somewhere either way) but each stage skips a 2.6GB
   copy. Opt-in: MPS has its own working-set limit and this has not been
   measured on a 24GB machine.
"""

import gc
from typing import Dict, Iterable, List

ALWAYS = ["sparse_structure_flow_model", "sparse_structure_decoder",
          "shape_slat_decoder", "tex_slat_decoder"]

_BY_TYPE: Dict[str, List[str]] = {
    "512": ["shape_slat_flow_model_512", "tex_slat_flow_model_512"],
    "1024": ["shape_slat_flow_model_1024", "tex_slat_flow_model_1024"],
    "1024_cascade": ["shape_slat_flow_model_512", "shape_slat_flow_model_1024",
                     "tex_slat_flow_model_1024"],
    "1536_cascade": ["shape_slat_flow_model_512", "shape_slat_flow_model_1024",
                     "tex_slat_flow_model_1024"],
}

PIPELINE_TYPES = tuple(_BY_TYPE)

# Approximate checkpoint sizes (bytes on disk, bf16/fp16) for reporting.
APPROX_BYTES = {
    "sparse_structure_flow_model": 2_584_426_920,
    "sparse_structure_decoder": 50_000_000,
    "shape_slat_decoder": 948_490_494,
    "tex_slat_decoder": 948_458_812,
    "shape_slat_flow_model_512": 2_584_574_424,
    "shape_slat_flow_model_1024": 2_584_574_424,
    "tex_slat_flow_model_512": 2_584_672_728,
    "tex_slat_flow_model_1024": 2_584_672_728,
}


def models_for(pipeline_types: Iterable[str]) -> List[str]:
    """Checkpoint names needed by one or more pipeline types, in load order."""
    names = list(ALWAYS)
    for t in pipeline_types:
        if t not in _BY_TYPE:
            raise ValueError(f"unknown pipeline type {t!r}; expected one of {PIPELINE_TYPES}")
        for n in _BY_TYPE[t]:
            if n not in names:
                names.append(n)
    return names


def approx_gb(names: Iterable[str]) -> float:
    return sum(APPROX_BYTES.get(n, 0) for n in names) / 1e9


DECODE_ONLY = ["shape_slat_decoder", "tex_slat_decoder"]


class _CondStub:
    """Stands in for DINOv3 / BiRefNet when a process only decodes."""

    def __init__(self, *a, **k):
        pass

    def to(self, *a, **k):
        return self

    def cpu(self):
        return self

    def __call__(self, *a, **k):
        raise RuntimeError("conditioning models were not loaded (decode-only pipeline)")


def _pipeline_args(path, config_file="pipeline.json"):
    import json
    import os
    if os.path.exists(f"{path}/{config_file}"):
        cfg = f"{path}/{config_file}"
    else:
        from huggingface_hub import hf_hub_download
        cfg = hf_hub_download(path, config_file)
    with open(cfg) as f:
        return json.load(f)["args"]


def load_pipeline(pipeline_types, path="microsoft/TRELLIS.2-4B", resident=False, decode_only=False):
    """Load only the checkpoints `pipeline_types` need.

    Uses the upstream `model_names_to_load` hook on the pipeline class so the
    base loader skips the rest. The class attribute is restored afterwards so
    other callers in the same process see upstream behaviour.

    decode_only=True loads just the two decoders (~1.9 GB) and stubs the
    conditioning models, for a process that decodes saved latents.
    """
    from trellis2.pipelines.trellis2_image_to_3d import Trellis2ImageTo3DPipeline

    if isinstance(pipeline_types, str):
        pipeline_types = [pipeline_types]
    names = list(DECODE_ONLY) if decode_only else models_for(pipeline_types)
    original = Trellis2ImageTo3DPipeline.model_names_to_load
    Trellis2ImageTo3DPipeline.model_names_to_load = names
    restore = []
    try:
        if decode_only:
            from trellis2.modules import image_feature_extractor
            from trellis2.pipelines import rembg
            args = _pipeline_args(path)
            for mod, key in ((image_feature_extractor, "image_cond_model"), (rembg, "rembg_model")):
                cls_name = args[key]["name"]
                restore.append((mod, cls_name, getattr(mod, cls_name, None)))
                setattr(mod, cls_name, _CondStub)
        pipeline = Trellis2ImageTo3DPipeline.from_pretrained(path)
    finally:
        Trellis2ImageTo3DPipeline.model_names_to_load = original
        for mod, cls_name, orig in restore:
            if orig is None:
                delattr(mod, cls_name)
            else:
                setattr(mod, cls_name, orig)
    if decode_only:
        pipeline.image_cond_model = None
        pipeline.rembg_model = None
    pipeline.low_vram = not resident
    pipeline._trellis3_path = path
    return pipeline


def ensure_models(pipeline, pipeline_type):
    """Lazily load any checkpoint a new pipeline type needs (server mode)."""
    from trellis2 import models as model_lib

    needed = models_for([pipeline_type])
    missing = [n for n in needed if n not in pipeline.models]
    if not missing:
        return []
    spec = pipeline._pretrained_args["models"]
    path = getattr(pipeline, "_trellis3_path", "microsoft/TRELLIS.2-4B")
    for n in missing:
        try:
            m = model_lib.from_pretrained(f"{path}/{spec[n]}")
        except Exception:
            m = model_lib.from_pretrained(spec[n])
        m.eval()
        if not pipeline.low_vram:
            m.to(pipeline.device)
        pipeline.models[n] = m
    return missing


def free_device_cache():
    """Release cached allocator blocks on whichever backend is active."""
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


STAGE_METHODS = ("sample_sparse_structure", "sample_shape_slat",
                 "sample_shape_slat_cascade", "sample_tex_slat",
                 "decode_shape_slat", "decode_tex_slat")


def install_stage_hooks(pipeline, after=free_device_cache):
    """Wrap each sampling/decoding stage so `after()` runs when it returns.

    Also tells the memory profiler (if one is active) which stage is running.
    Idempotent: a pipeline already hooked is left alone.
    """
    if getattr(pipeline, "_trellis3_hooked", False):
        return pipeline
    from trellis3 import profile as _profile
    for name in STAGE_METHODS:
        fn = getattr(pipeline, name, None)
        if fn is None:
            continue

        def wrapped(*a, _fn=fn, _name=name, **kw):
            _profile.mark(_name)
            out = _fn(*a, **kw)
            after()
            _profile.mark(f"after:{_name}")
            return out

        setattr(pipeline, name, wrapped)
    pipeline._trellis3_hooked = True
    return pipeline


def parse_vm_stat(text, page_size=16384):
    """Free + inactive + speculative pages from macOS `vm_stat` output, in GB."""
    import re
    m = re.search(r"page size of (\d+) bytes", text or "")
    if m:
        page_size = int(m.group(1))
    total = 0
    found = False
    for key in ("Pages free", "Pages inactive", "Pages speculative"):
        m = re.search(rf"{key}:\s+(\d+)", text or "")
        if m:
            total += int(m.group(1))
            found = True
    return total * page_size / 1e9 if found else None


def parse_meminfo(text):
    """MemAvailable from /proc/meminfo, in GB."""
    import re
    m = re.search(r"MemAvailable:\s+(\d+) kB", text or "")
    return int(m.group(1)) / 1e6 if m else None


def free_memory_gb():
    """Best-effort available system memory in GB, or None."""
    try:
        import psutil
        return psutil.virtual_memory().available / 1e9
    except Exception:
        pass
    import subprocess
    import sys
    try:
        if sys.platform == "darwin":
            out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5).stdout
            return parse_vm_stat(out)
        with open("/proc/meminfo") as f:
            return parse_meminfo(f.read())
    except Exception:
        return None


def rss_gb():
    """Resident set size of this process in GB, or None if unavailable."""
    try:
        import resource
        import sys
        kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macOS reports bytes, Linux reports kilobytes.
        return kb / 1e9 if sys.platform == "darwin" else kb / 1e6
    except Exception:
        return None
