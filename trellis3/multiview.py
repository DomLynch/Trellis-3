"""
Multi-image conditioning for the staged sampler.

Ported from microsoft/TRELLIS.2 pull request 104 ("multi-image conditioning").
Upstream conditions on one image: cond has shape (1, tokens, C). Here cond is
the concatenation of one DINOv3 feature set per view, shape (N, tokens, C),
and the sampler's `_inference_model` is wrapped so each denoising step uses
the views as follows:

  stochastic      step k uses view (k mod N). One model call per step, same
                  cost as a single image, but each step sees only one view.
  multidiffusion  every step averages the velocity predicted for every view
                  (N model calls per step), then applies classifier-free
                  guidance once. Slower, much more consistent.

The negative condition stays a single zero tensor. With N == 1 nothing is
wrapped, so the single-image path is untouched.
"""

from contextlib import ExitStack, contextmanager

FUSION_MODES = ("multidiffusion", "stochastic")
DEFAULT_FUSION = "multidiffusion"

# stage -> sampler attribute on the pipeline
SAMPLER_FOR_STAGE = {"coords": "sparse_structure_sampler",
                     "shape": "shape_slat_sampler",
                     "tex": "tex_slat_sampler"}


def stack_conds(conds):
    """[{'cond': (1,T,C), 'neg_cond': (1,T,C)}, ...] -> {'cond': (N,T,C), 'neg_cond': (1,T,C)}."""
    import torch
    if not conds:
        raise ValueError("need at least one conditioning dict")
    stacked = torch.cat([c["cond"] for c in conds], dim=0)
    return {"cond": stacked, "neg_cond": torch.zeros_like(stacked[:1])}


def get_cond_multi(pipeline, images, resolution):
    """Conditioning for one or many preprocessed images at one DINO resolution.
    A single image returns exactly what pipeline.get_cond returns."""
    if len(images) == 1:
        return pipeline.get_cond(list(images), resolution)
    return stack_conds([pipeline.get_cond([im], resolution) for im in images])


def _stochastic(old, num_images):
    counter = {"n": 0}

    def inference(self, model, x_t, t, cond, **kwargs):
        i = counter["n"] % num_images
        counter["n"] += 1
        return old(model, x_t, t, cond=cond[i:i + 1], **kwargs)
    return inference


def _multidiffusion(flow_euler_inference):
    def inference(self, model, x_t, t, cond, neg_cond, guidance_strength, guidance_interval,
                  guidance_rescale=0.0, **kwargs):
        preds = [flow_euler_inference(self, model, x_t, t, cond[i:i + 1], **kwargs)
                 for i in range(len(cond))]
        pred = sum(preds) / len(preds)
        if not (guidance_interval[0] <= t <= guidance_interval[1]):
            return pred
        neg_pred = flow_euler_inference(self, model, x_t, t, neg_cond, **kwargs)
        pred_cfg = guidance_strength * pred + (1 - guidance_strength) * neg_pred
        if guidance_rescale > 0:
            x_0_pos = self._pred_to_xstart(x_t, t, pred)
            x_0_cfg = self._pred_to_xstart(x_t, t, pred_cfg)
            dims = list(range(1, x_0_pos.ndim))
            x_0_resc = x_0_cfg * (x_0_pos.std(dim=dims, keepdim=True) / x_0_cfg.std(dim=dims, keepdim=True))
            x_0 = guidance_rescale * x_0_resc + (1 - guidance_rescale) * x_0_cfg
            pred_cfg = self._xstart_to_pred(x_t, t, x_0)
        return pred_cfg
    return inference


@contextmanager
def inject_sampler(sampler, num_images, mode=DEFAULT_FUSION, base=None):
    """Temporarily replace `sampler._inference_model` with a multi-view version.
    `base` is the unguided single-view model call (FlowEulerSampler._inference_model);
    tests pass a stand-in so upstream need not be importable."""
    if mode not in FUSION_MODES:
        raise ValueError(f"unsupported fusion mode {mode!r}; expected one of {FUSION_MODES}")
    if num_images <= 1:
        yield sampler
        return
    old = sampler._inference_model          # bound method of the full mixin chain
    if mode == "stochastic":
        new = _stochastic(old, num_images)
    else:
        if base is None:
            from trellis2.pipelines.samplers import FlowEulerSampler
            base = FlowEulerSampler._inference_model
        new = _multidiffusion(base)
    sampler._inference_model = new.__get__(sampler, type(sampler))
    try:
        yield sampler
    finally:
        # the wrapper lives in the instance dict; dropping it restores the class method
        sampler.__dict__.pop("_inference_model", None)


@contextmanager
def inject_pipeline(pipeline, num_images, mode=DEFAULT_FUSION):
    """Wrap all three samplers of a pipeline for the duration of the block."""
    with ExitStack() as stack:
        for attr in SAMPLER_FOR_STAGE.values():
            s = getattr(pipeline, attr, None)
            if s is not None:
                stack.enter_context(inject_sampler(s, num_images, mode))
        yield pipeline
