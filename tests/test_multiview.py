import pytest
import torch

from trellis3 import multiview


class _Sampler:
    """Stand-in for the CFG + interval sampler chain."""

    def __init__(self):
        self.seen = []

    def _inference_model(self, model, x_t, t, cond, neg_cond=None, guidance_strength=1.0,
                         guidance_interval=(0, 1), **kw):
        self.seen.append(cond.clone())
        return cond.mean() * torch.ones_like(x_t)

    def _pred_to_xstart(self, x_t, t, p):
        return x_t - p

    def _xstart_to_pred(self, x_t, t, x0):
        return x_t - x0


def _base(self, model, x_t, t, cond, **kw):
    return cond.mean() * torch.ones_like(x_t)


def _cond(values, n=3, c=4):
    return torch.cat([torch.full((1, n, c), float(v)) for v in values])


def test_stack_conds_shapes():
    one = {"cond": torch.ones(1, 5, 8), "neg_cond": torch.zeros(1, 5, 8)}
    two = {"cond": 2 * torch.ones(1, 5, 8), "neg_cond": torch.zeros(1, 5, 8)}
    out = multiview.stack_conds([one, two, one, two])
    assert out["cond"].shape == (4, 5, 8)
    assert out["neg_cond"].shape == (1, 5, 8)
    assert out["neg_cond"].abs().sum() == 0


def test_get_cond_multi_single_is_upstream():
    class P:
        def get_cond(self, imgs, res):
            return {"cond": torch.ones(len(imgs), 2, 2), "neg_cond": torch.zeros(len(imgs), 2, 2)}
    out = multiview.get_cond_multi(P(), ["a"], 512)
    assert out["cond"].shape == (1, 2, 2) and out["neg_cond"].shape == (1, 2, 2)
    out = multiview.get_cond_multi(P(), ["a", "b", "c"], 512)
    assert out["cond"].shape == (3, 2, 2) and out["neg_cond"].shape == (1, 2, 2)


def test_single_image_does_not_wrap():
    s = _Sampler()
    with multiview.inject_sampler(s, 1, "multidiffusion"):
        assert "_inference_model" not in s.__dict__


def test_stochastic_cycles_views_and_restores():
    s = _Sampler()
    cond = _cond([1, 2, 3])
    x = torch.zeros(1, 2)
    with multiview.inject_sampler(s, 3, "stochastic"):
        outs = [s._inference_model(None, x, 0.5, cond=cond, neg_cond=cond[:1]).mean().item() for _ in range(5)]
    assert outs == [1, 2, 3, 1, 2]
    assert all(c.shape[0] == 1 for c in s.seen)
    assert "_inference_model" not in s.__dict__


def test_multidiffusion_averages_views_inside_interval():
    s = _Sampler()
    cond = _cond([1, 3])
    neg = torch.zeros(1, 3, 4)
    x = torch.zeros(1, 2)
    with multiview.inject_sampler(s, 2, "multidiffusion", base=_base):
        out = s._inference_model(None, x, 0.5, cond=cond, neg_cond=neg, guidance_strength=3.0,
                                 guidance_interval=(0.0, 1.0))
    # pos = mean(1,3) = 2, neg = 0, cfg = 3*2 + (1-3)*0 = 6
    assert out.mean().item() == pytest.approx(6.0)


def test_multidiffusion_outside_interval_skips_guidance():
    s = _Sampler()
    cond = _cond([1, 3])
    x = torch.zeros(1, 2)
    with multiview.inject_sampler(s, 2, "multidiffusion", base=_base):
        out = s._inference_model(None, x, 0.9, cond=cond, neg_cond=torch.zeros(1, 3, 4),
                                 guidance_strength=3.0, guidance_interval=(0.0, 0.5))
    assert out.mean().item() == pytest.approx(2.0)


def test_multidiffusion_equal_views_equals_single():
    """N identical views must reproduce the single-view CFG result."""
    s = _Sampler()
    x = torch.zeros(1, 2)
    neg = torch.zeros(1, 3, 4)
    with multiview.inject_sampler(s, 3, "multidiffusion", base=_base):
        multi = s._inference_model(None, x, 0.5, cond=_cond([2, 2, 2]), neg_cond=neg,
                                   guidance_strength=3.0, guidance_interval=(0, 1))
    assert multi.mean().item() == pytest.approx(6.0)


def test_bad_mode():
    with pytest.raises(ValueError):
        with multiview.inject_sampler(_Sampler(), 2, "nope"):
            pass
