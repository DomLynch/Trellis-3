"""
Resumable, stage-by-stage sampling that mirrors `Trellis2ImageTo3DPipeline.run`
up to (not including) decode.

Why this exists: on a 24 GB Mac the 1024 cascade samples for ~20 minutes and
then dies at decode. With `run()` that is all one call — the sampler weights
are still resident when the decoder starts, and a decode failure throws the
sampling away. Here each stage writes its result into a `state` dict that the
caller can save (`latents.save_state`), the models a stage no longer needs can
be dropped (`release=True`), a later process can resume or decode from the
saved state, and an in-process memory guard can stop cleanly between steps.

Stage order and arguments are copied from upstream `run()`:
    preprocess -> seed -> cond(512[,1024]) -> coords -> shape -> tex
so a straight (non-resumed) run is numerically identical to `pipeline.run`.
RNG state is saved after every stage, so a resume continues the same stream.
"""

import gc
import io
import time

STAGES = ("coords", "shape", "tex")

_SS_RES = {"512": 32, "1024": 64, "1024_cascade": 32, "1536_cascade": 32}

# models a stage needs; everything else can be released once the stage is done
_NEEDS = {
    "coords": ["sparse_structure_flow_model", "sparse_structure_decoder"],
    "shape": {"512": ["shape_slat_flow_model_512"],
              "1024": ["shape_slat_flow_model_1024"],
              "1024_cascade": ["shape_slat_flow_model_512", "shape_slat_flow_model_1024", "shape_slat_decoder"],
              "1536_cascade": ["shape_slat_flow_model_512", "shape_slat_flow_model_1024", "shape_slat_decoder"]},
    "tex": {"512": ["tex_slat_flow_model_512"], "1024": ["tex_slat_flow_model_1024"],
            "1024_cascade": ["tex_slat_flow_model_1024"], "1536_cascade": ["tex_slat_flow_model_1024"]},
}


class MemoryPressure(RuntimeError):
    """Raised between sampler steps when free memory drops under the floor."""


def _steps_for(steps, stage):
    """steps may be None, an int (all stages) or a dict {ss, shape, tex}."""
    if steps is None:
        return {}
    if isinstance(steps, int):
        return {"steps": steps}
    key = {"coords": "ss", "shape": "shape", "tex": "tex"}[stage]
    v = steps.get(key, steps.get("all"))
    return {"steps": v} if v else {}


def _rng_snapshot():
    import torch
    snap = {"cpu": torch.get_rng_state()}
    try:
        if torch.backends.mps.is_available():
            snap["mps"] = torch.mps.get_rng_state()
    except Exception:
        pass
    return snap


def _rng_restore(snap):
    import torch
    if not snap:
        return
    torch.set_rng_state(snap["cpu"])
    if "mps" in snap:
        try:
            torch.mps.set_rng_state(snap["mps"])
        except Exception:
            pass


def _release(pipeline, names, log):
    freed = []
    for n in names:
        if n in pipeline.models:
            del pipeline.models[n]
            freed.append(n)
    if freed:
        gc.collect()
        from trellis3.memory import free_device_cache
        free_device_cache()
        log(f"  released {', '.join(freed)}")
    return freed


def _release_cond(pipeline, log):
    if getattr(pipeline, "image_cond_model", None) is not None or getattr(pipeline, "rembg_model", None) is not None:
        pipeline.image_cond_model = None
        pipeline.rembg_model = None
        gc.collect()
        from trellis3.memory import free_device_cache
        free_device_cache()
        log("  released image_cond_model, rembg_model")


class _Guard:
    """Wraps the three samplers' sample_once so a memory check runs before
    every step. The probe is a vm_stat / psutil read (milliseconds) against
    steps that take seconds to a minute, so it is checked synchronously."""

    def __init__(self, pipeline, min_free_gb, probe=None):
        from trellis3.memory import free_memory_gb
        self.pipeline = pipeline
        self.min_free_gb = min_free_gb
        self.probe = probe or free_memory_gb
        self._orig = []

    def check(self):
        free = self.probe()
        if free is not None and free < self.min_free_gb:
            raise MemoryPressure(f"free memory {free:.1f} GB < floor {self.min_free_gb} GB")

    def __enter__(self):
        if not self.min_free_gb:
            return self
        for attr in ("sparse_structure_sampler", "shape_slat_sampler", "tex_slat_sampler"):
            s = getattr(self.pipeline, attr, None)
            if s is None or not hasattr(s, "sample_once"):
                continue
            orig = s.sample_once

            def guarded(*a, _orig=orig, **kw):
                self.check()
                return _orig(*a, **kw)

            s.sample_once = guarded
            self._orig.append((s, orig))
        return self

    def __exit__(self, *exc):
        for s, orig in self._orig:
            s.sample_once = orig
        return False


def sample_stages(pipeline, image, pipeline_type, seed=42, steps=None, max_num_tokens=49152,
                  state=None, on_stage=None, release=False, min_free_gb=0.0, probe=None, log=print):
    """Run the sampling stages, resuming from `state` if it has completed ones.

    Returns the state dict: pipeline_type, seed, image_png (preprocessed),
    coords, shape_feats, shape_coords, res, tex_feats, done (list), rng.
    `on_stage(name, state)` is called after each completed stage.
    """
    import torch
    from PIL import Image

    if pipeline_type not in _SS_RES:
        raise ValueError(f"Invalid pipeline type: {pipeline_type}")
    for n in _NEEDS["coords"] + _NEEDS["shape"][pipeline_type] + _NEEDS["tex"][pipeline_type]:
        assert n in pipeline.models, f"missing model {n} for pipeline type {pipeline_type}"

    state = dict(state or {})
    done = list(state.get("done", []))
    if state and (state.get("pipeline_type") != pipeline_type or state.get("seed") != seed):
        raise ValueError("state was produced with a different pipeline_type/seed; refusing to resume")
    state.update(pipeline_type=pipeline_type, seed=seed)

    def finish(stage):
        done.append(stage)
        state["done"] = list(done)
        state["rng"] = _rng_snapshot()
        if on_stage:
            on_stage(stage, state)

    # --- preprocess + seed (identical order to upstream run) -----------------
    if "image_png" in state:
        img = Image.open(io.BytesIO(state["image_png"]))
        img.load()
    else:
        img = pipeline.preprocess_image(image)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        state["image_png"] = buf.getvalue()
    if done:
        _rng_restore(state.get("rng"))
    else:
        torch.manual_seed(seed)

    cond_512 = pipeline.get_cond([img], 512)
    cond_1024 = pipeline.get_cond([img], 1024) if pipeline_type != "512" else None
    if release:
        _release_cond(pipeline, log)

    with _Guard(pipeline, min_free_gb, probe=probe):
        # --- coords ------------------------------------------------------------
        if "coords" not in done:
            t0 = time.time()
            coords = pipeline.sample_sparse_structure(cond_512, _SS_RES[pipeline_type], 1,
                                                      _steps_for(steps, "coords"))
            state["coords"] = coords
            log(f"  stage coords: {coords.shape[0]:,} voxels ({time.time() - t0:.0f}s)")
            finish("coords")
        coords = state["coords"]
        if release:
            _release(pipeline, _NEEDS["coords"], log)

        # --- shape -------------------------------------------------------------
        if "shape" not in done:
            t0 = time.time()
            sp = _steps_for(steps, "shape")
            if pipeline_type == "512":
                shape_slat = pipeline.sample_shape_slat(cond_512, pipeline.models["shape_slat_flow_model_512"],
                                                        coords, sp)
                res = 512
            elif pipeline_type == "1024":
                shape_slat = pipeline.sample_shape_slat(cond_1024, pipeline.models["shape_slat_flow_model_1024"],
                                                        coords, sp)
                res = 1024
            else:
                target = 1024 if pipeline_type == "1024_cascade" else 1536
                shape_slat, res = pipeline.sample_shape_slat_cascade(
                    cond_512, cond_1024,
                    pipeline.models["shape_slat_flow_model_512"], pipeline.models["shape_slat_flow_model_1024"],
                    512, target, coords, sp, max_num_tokens)
            state["shape_feats"] = shape_slat.feats
            state["shape_coords"] = shape_slat.coords
            state["res"] = int(res)
            log(f"  stage shape: {shape_slat.feats.shape[0]:,} tokens at {res} ({time.time() - t0:.0f}s)")
            finish("shape")
        if release:
            keep = set(_NEEDS["tex"][pipeline_type]) | {"shape_slat_decoder", "tex_slat_decoder"}
            _release(pipeline, [n for n in _NEEDS["shape"][pipeline_type] if n not in keep], log)

        # --- tex ---------------------------------------------------------------
        if "tex" not in done:
            t0 = time.time()
            shape_slat = _make_sparse(state["shape_feats"], state["shape_coords"], pipeline.device)
            cond = cond_512 if pipeline_type == "512" else cond_1024
            name = _NEEDS["tex"][pipeline_type][0]
            tex_slat = pipeline.sample_tex_slat(cond, pipeline.models[name], shape_slat, _steps_for(steps, "tex"))
            state["tex_feats"] = tex_slat.feats
            log(f"  stage tex: done ({time.time() - t0:.0f}s)")
            finish("tex")
        if release:
            _release(pipeline, _NEEDS["tex"][pipeline_type], log)
    return state


def _make_sparse(feats, coords, device):
    from trellis2.modules.sparse import SparseTensor
    return SparseTensor(feats=feats.to(device), coords=coords.to(device))


def decode_state(pipeline, state, log=print):
    """Decode a completed state into a MeshWithVoxel (upstream decode_latent)."""
    for k in ("shape_feats", "shape_coords", "tex_feats", "res"):
        if k not in state:
            raise ValueError(f"state is incomplete: missing {k} (done={state.get('done')})")
    t0 = time.time()
    shape_slat = _make_sparse(state["shape_feats"], state["shape_coords"], pipeline.device)
    tex_slat = shape_slat.replace(state["tex_feats"].to(pipeline.device))
    out = pipeline.decode_latent(shape_slat, tex_slat, state["res"])
    mesh = out[0] if isinstance(out, list) else out
    log(f"  decode: {mesh.vertices.shape[0]:,} vertices ({time.time() - t0:.0f}s)")
    return mesh
