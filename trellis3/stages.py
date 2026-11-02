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

from trellis3 import profile as _profile

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
                  state=None, on_stage=None, release=False, min_free_gb=0.0, probe=None, log=print,
                  fusion="multidiffusion"):
    """Run the sampling stages, resuming from `state` if it has completed ones.

    Returns the state dict: pipeline_type, seed, image_png (preprocessed),
    coords, shape_feats, shape_coords, res, tex_feats, done (list), rng.
    `on_stage(name, state)` is called after each completed stage.

    `image` may be a list of views (see trellis3.multiview); with more than one
    the per-view conditioning is fused by `fusion` (multidiffusion|stochastic).
    A single image takes the unchanged upstream path.
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
    from trellis3 import multiview
    images = list(image) if isinstance(image, (list, tuple)) else [image]
    if "image_pngs" in state:
        imgs = []
        for b in state["image_pngs"]:
            im = Image.open(io.BytesIO(b))
            im.load()
            imgs.append(im)
    elif "image_png" in state:
        imgs = [Image.open(io.BytesIO(state["image_png"]))]
        imgs[0].load()
    else:
        imgs = [pipeline.preprocess_image(im) for im in images]
        pngs = []
        for im in imgs:
            buf = io.BytesIO()
            im.save(buf, format="PNG")
            pngs.append(buf.getvalue())
        state["image_png"] = pngs[0]
        if len(pngs) > 1:
            state["image_pngs"] = pngs
            state["fusion"] = fusion
    n_views = len(imgs)
    if n_views > 1:
        fusion = state.get("fusion", fusion)
        log(f"  multi-image conditioning: {n_views} views, fusion={fusion}")
    if done:
        _rng_restore(state.get("rng"))
    else:
        torch.manual_seed(seed)

    cond_512 = multiview.get_cond_multi(pipeline, imgs, 512)
    cond_1024 = multiview.get_cond_multi(pipeline, imgs, 1024) if pipeline_type != "512" else None
    if release:
        _release_cond(pipeline, log)

    with _Guard(pipeline, min_free_gb, probe=probe), multiview.inject_pipeline(pipeline, n_views, fusion):
        # --- coords ------------------------------------------------------------
        if "coords" not in done:
            t0 = time.time()
            _profile.mark("coords")
            coords = pipeline.sample_sparse_structure(cond_512, _SS_RES[pipeline_type], 1,
                                                      _steps_for(steps, "coords"))
            state["coords"] = coords
            log(f"  stage coords: {coords.shape[0]:,} voxels ({time.time() - t0:.0f}s)")
            finish("coords")
        coords = state["coords"]
        if release:
            _release(pipeline, _NEEDS["coords"], log)

        # --- shape -------------------------------------------------------------
        sp = _steps_for(steps, "shape")
        if pipeline_type in ("512", "1024"):
            if "shape" not in done:
                t0 = time.time()
                _profile.mark("shape")
                if pipeline_type == "512":
                    shape_slat = pipeline.sample_shape_slat(cond_512, pipeline.models["shape_slat_flow_model_512"],
                                                            coords, sp)
                    res = 512
                else:
                    shape_slat = pipeline.sample_shape_slat(cond_1024, pipeline.models["shape_slat_flow_model_1024"],
                                                            coords, sp)
                    res = 1024
                state["shape_feats"] = shape_slat.feats
                state["shape_coords"] = shape_slat.coords
                state["res"] = int(res)
                log(f"  stage shape: {shape_slat.feats.shape[0]:,} tokens at {res} ({time.time() - t0:.0f}s)")
                finish("shape")
        else:
            # The cascade is three separate things upstream runs in one call:
            # a 512 sampling pass, the shape decoder upsampling it to the
            # high-res grid, and a 1024/1536 sampling pass. Each is its own
            # sub-stage here so the latents survive a crash in the next one
            # and the profiler can tell them apart.
            target = 1024 if pipeline_type == "1024_cascade" else 1536
            if "shape_lr" not in done:
                t0 = time.time()
                _profile.mark("shape_lr")
                slat = _sample_slat(pipeline, pipeline.models["shape_slat_flow_model_512"], cond_512, coords, sp,
                                    "Sampling shape SLat (512)")
                state["shape_lr_feats"], state["shape_lr_coords"] = slat.feats, slat.coords
                log(f"  stage shape_lr: {slat.feats.shape[0]:,} tokens at 512 ({time.time() - t0:.0f}s)")
                finish("shape_lr")
            if release:
                _release(pipeline, ["shape_slat_flow_model_512"], log)

            if "shape_up" not in done:
                t0 = time.time()
                _profile.mark("shape_up")
                slat = _make_sparse(state["shape_lr_feats"], state["shape_lr_coords"], pipeline.device)
                hr_coords, res = _upsample_coords(pipeline, slat, 512, target, max_num_tokens)
                state["shape_hr_coords"], state["res"] = hr_coords, int(res)
                log(f"  stage shape_up: {hr_coords.shape[0]:,} tokens at {res} ({time.time() - t0:.0f}s)")
                finish("shape_up")

            if "shape_hr" not in done:
                t0 = time.time()
                _profile.mark("shape_hr")
                slat = _sample_slat(pipeline, pipeline.models["shape_slat_flow_model_1024"], cond_1024,
                                    state["shape_hr_coords"], sp, f"Sampling shape SLat ({state['res']})")
                state["shape_feats"], state["shape_coords"] = slat.feats, slat.coords
                log(f"  stage shape_hr: {slat.feats.shape[0]:,} tokens at {state['res']} ({time.time() - t0:.0f}s)")
                finish("shape_hr")
                finish("shape")
        if release:
            keep = set(_NEEDS["tex"][pipeline_type]) | {"shape_slat_decoder", "tex_slat_decoder"}
            _release(pipeline, [n for n in _NEEDS["shape"][pipeline_type] if n not in keep], log)

        # --- tex ---------------------------------------------------------------
        if "tex" not in done:
            t0 = time.time()
            _profile.mark("tex")
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


def _sample_slat(pipeline, flow_model, cond, coords, sampler_params, desc):
    """One shape-SLat sampling pass, exactly as upstream sample_shape_slat."""
    import torch
    from trellis2.modules.sparse import SparseTensor
    noise = SparseTensor(
        feats=torch.randn(coords.shape[0], flow_model.in_channels).to(pipeline.device),
        coords=coords.to(pipeline.device),
    )
    params = {**pipeline.shape_slat_sampler_params, **sampler_params}
    if pipeline.low_vram:
        flow_model.to(pipeline.device)
    slat = pipeline.shape_slat_sampler.sample(flow_model, noise, **cond, **params,
                                              verbose=True, tqdm_desc=desc).samples
    if pipeline.low_vram:
        flow_model.cpu()
    std = torch.tensor(pipeline.shape_slat_normalization["std"])[None].to(slat.device)
    mean = torch.tensor(pipeline.shape_slat_normalization["mean"])[None].to(slat.device)
    return slat * std + mean


def _upsample_coords(pipeline, slat, lr_resolution, resolution, max_num_tokens):
    """Shape-decoder upsample + token-budget quantisation, as upstream cascade."""
    import torch
    dec = pipeline.models["shape_slat_decoder"]
    if pipeline.low_vram:
        dec.to(pipeline.device)
        dec.low_vram = True
    hr_coords = dec.upsample(slat, upsample_times=4)
    if pipeline.low_vram:
        dec.cpu()
        dec.low_vram = False
    hr_resolution = resolution
    while True:
        quant = torch.cat([hr_coords[:, :1],
                           ((hr_coords[:, 1:] + 0.5) / lr_resolution * (hr_resolution // 16)).int()], dim=1)
        coords = quant.unique(dim=0)
        if coords.shape[0] < max_num_tokens or hr_resolution == 1024:
            if hr_resolution != resolution:
                print(f"Due to the limited number of tokens, the resolution is reduced to {hr_resolution}.")
            break
        hr_resolution -= 128
    return coords, hr_resolution


def _make_sparse(feats, coords, device):
    from trellis2.modules.sparse import SparseTensor
    return SparseTensor(feats=feats.to(device), coords=coords.to(device))


def _peak_gb():
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1e9
    except Exception:
        pass
    return None


def _offload_state(state):
    """Move the state's device tensors (voxel coords, shape and texture latents) to the CPU. The decode builds its own device copies, so
    the state's would otherwise stay resident next to them for the whole decode (microsoft/TRELLIS.2 issue 188: the sampling tensors are
    still held when mesh post-processing starts). Values are unchanged; `_make_sparse` moves them back on demand."""
    moved = []
    for k, v in list(state.items()):
        try:
            import torch
            if isinstance(v, torch.Tensor) and v.device.type != "cpu":
                state[k] = v.cpu()
                moved.append(k)
        except Exception:
            pass
    return moved


def _decode_lean(pipeline, slats, resolution, attrs_to_cpu=True):
    """Shape decode, texture decode, release, hole filling. `slats` is [shape_slat, tex_slat] and is emptied here, so the caller keeps no reference
    and the tensors really are freed before `Mesh.fill_holes()` builds its adjacency structures."""
    import gc
    import torch
    from trellis2.representations import MeshWithVoxel
    from trellis3.memory import free_device_cache
    shape_slat, tex_slat = slats
    slats.clear()
    meshes, subs = pipeline.decode_shape_slat(shape_slat, resolution)
    shape_slat = None
    tex_voxels = pipeline.decode_tex_slat(tex_slat, subs)
    tex_slat = subs = None
    gc.collect()
    free_device_cache()
    out = []
    for m, v in zip(meshes, tex_voxels):
        m.fill_holes()
        # The attribute volume (voxel colours) is only needed at the bake: keep it, and its coordinates, off the device until then.
        attrs, coords = (v.feats.cpu(), v.coords[:, 1:].cpu()) if attrs_to_cpu else (v.feats, v.coords[:, 1:])
        out.append(MeshWithVoxel(m.vertices, m.faces, origin=[-0.5, -0.5, -0.5], voxel_size=1 / resolution,
                                 coords=coords, attrs=attrs,
                                 voxel_shape=torch.Size([*v.shape, *v.spatial_shape]),
                                 layout=pipeline.pbr_attr_layout))
    meshes = tex_voxels = None
    gc.collect()
    free_device_cache()
    return out


def decode_latent_lean(pipeline, shape_slat, tex_slat, resolution, attrs_to_cpu=True):
    """pipeline.decode_latent with the sampling data released before mesh post-processing.

    Same maths and the same mesh as upstream decode_latent. Upstream keeps the shape latent, the texture latent
    and the decoder's sub-structures alive while `Mesh.fill_holes()` builds its adjacency structures, so the peak
    is reached there. Releasing them first lowers the peak (reported 21 -> 11 GB on a 23 GB card in
    microsoft/TRELLIS.2 issue 188, first reported by rpep) without changing the mesh. Re-implemented from the
    issue text. Also keeps the attribute volume on the CPU until the bake (`attrs_to_cpu`). Callers that hold their own
    references to the latents (this wrapper's arguments) free nothing until they drop them; `decode_state` passes ownership.
    """
    return _decode_lean(pipeline, [shape_slat, tex_slat], resolution, attrs_to_cpu)


def decode_state(pipeline, state, log=print, lean=True, tile_voxels=None, attrs_to_cpu=True):
    """Decode a completed state into a MeshWithVoxel (upstream decode_latent, lean by default).

    tile_voxels: run the shape decoder's last two stages in spatial tiles of this many voxels per side (0 = off; None reads
    TRELLIS3_TILE_VOXELS, default off); see trellis3.decoder_tiling."""
    for k in ("shape_feats", "shape_coords", "tex_feats", "res"):
        if k not in state:
            raise ValueError(f"state is incomplete: missing {k} (done={state.get('done')})")
    t0 = time.time()
    if lean and hasattr(pipeline, "decode_shape_slat") and hasattr(pipeline, "decode_tex_slat"):
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass
        dec = (getattr(pipeline, "models", None) or {}).get("shape_slat_decoder")
        if dec is not None:
            from trellis3 import decoder_tiling
            tile = decoder_tiling.install(dec, tile_voxels)
            if tile:
                log(f"  shape decoder: last two stages in {tile}-voxel tiles")
        slats = [_make_sparse(state["shape_feats"], state["shape_coords"], pipeline.device), None]
        slats[1] = slats[0].replace(state["tex_feats"].to(pipeline.device))
        moved = _offload_state(state)
        if moved:
            log(f"  state tensors moved to the CPU before decoding: {', '.join(moved)}")
        out = _decode_lean(pipeline, slats, state["res"], attrs_to_cpu)
        if dec is not None and getattr(dec, "last_tiles", None):
            log(f"  tiled decode: {dec.last_tiles} tiles")
    else:
        shape_slat = _make_sparse(state["shape_feats"], state["shape_coords"], pipeline.device)
        tex_slat = shape_slat.replace(state["tex_feats"].to(pipeline.device))
        out = pipeline.decode_latent(shape_slat, tex_slat, state["res"])
    mesh = out[0] if isinstance(out, list) else out
    peak = _peak_gb()
    log(f"  decode: {mesh.vertices.shape[0]:,} vertices ({time.time() - t0:.0f}s)"
        + (f", peak GPU {peak:.1f} GB" if peak else ""))
    return mesh
