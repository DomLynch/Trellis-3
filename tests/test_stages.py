import sys
import types

import pytest
import torch
from PIL import Image

from trellis3 import stages


class _Sparse:
    def __init__(self, feats, coords):
        self.feats, self.coords = feats, coords

    def replace(self, feats):
        return _Sparse(feats, self.coords)


class _Sampler:
    def __init__(self):
        self.calls = 0

    def sample_once(self, *a, **k):
        self.calls += 1
        return None


class _Pipe:
    """Records every stage call with its arguments."""

    def __init__(self, ptype):
        from trellis3 import memory
        self.models = {n: object() for n in memory.models_for([ptype])}
        self.device = "cpu"
        self.image_cond_model = object()
        self.rembg_model = object()
        self.sparse_structure_sampler = _Sampler()
        self.shape_slat_sampler = _Sampler()
        self.tex_slat_sampler = _Sampler()
        self.calls = []
        self.fail_at = None

    def preprocess_image(self, img):
        self.calls.append(("preprocess",))
        return img.convert("RGB")

    def get_cond(self, imgs, res):
        self.calls.append(("cond", res))
        return {"cond": res}

    def _step(self, sampler):
        sampler.sample_once()

    def sample_sparse_structure(self, cond, res, n, params):
        self.calls.append(("ss", cond["cond"], res, params))
        if self.fail_at == "coords":
            raise stages.MemoryPressure("boom")
        self._step(self.sparse_structure_sampler)
        return torch.randint(0, 32, (100, 4), dtype=torch.int32)

    def sample_shape_slat(self, cond, model, coords, params):
        self.calls.append(("shape", cond["cond"], params))
        self._step(self.shape_slat_sampler)
        return _Sparse(torch.randn(100, 32), coords)

    def sample_shape_slat_cascade(self, lr_cond, cond, m_lr, m, lr_res, res, coords, params, max_tokens):
        self.calls.append(("shape_cascade", lr_cond["cond"], cond["cond"], lr_res, res, params, max_tokens))
        if self.fail_at == "shape":
            raise stages.MemoryPressure("boom")
        self._step(self.shape_slat_sampler)
        return _Sparse(torch.randn(300, 32), torch.randint(0, 64, (300, 4), dtype=torch.int32)), res

    def sample_tex_slat(self, cond, model, shape_slat, params):
        self.calls.append(("tex", cond["cond"], params, shape_slat.feats.shape[0]))
        self._step(self.tex_slat_sampler)
        return shape_slat.replace(torch.randn(shape_slat.feats.shape[0], 32))

    def decode_latent(self, shape_slat, tex_slat, res):
        self.calls.append(("decode", res, shape_slat.feats.shape, tex_slat.feats.shape))
        m = types.SimpleNamespace(vertices=torch.zeros((3, 3)), faces=torch.tensor([[0, 1, 2]]))
        return [m]


@pytest.fixture(autouse=True)
def fake_sparse(monkeypatch):
    mod = types.ModuleType("trellis2.modules.sparse")
    mod.SparseTensor = _Sparse
    monkeypatch.setitem(sys.modules, "trellis2", types.ModuleType("trellis2"))
    monkeypatch.setitem(sys.modules, "trellis2.modules", types.ModuleType("trellis2.modules"))
    monkeypatch.setitem(sys.modules, "trellis2.modules.sparse", mod)


@pytest.fixture
def img():
    return Image.new("RGBA", (8, 8), (10, 20, 30, 255))


def test_512_order_matches_upstream_run(img):
    p = _Pipe("512")
    seen = []
    st = stages.sample_stages(p, img, "512", seed=5, on_stage=lambda n, s: seen.append(n), log=lambda *_: None)
    kinds = [c[0] for c in p.calls]
    assert kinds == ["preprocess", "cond", "ss", "shape", "tex"]
    assert p.calls[1][1] == 512 and p.calls[2][2] == 32          # cond 512 only, ss_res 32
    assert seen == ["coords", "shape", "tex"] and st["done"] == ["coords", "shape", "tex"]
    assert st["res"] == 512 and "image_png" in st and "rng" in st
    assert set(p.models) == {"sparse_structure_flow_model", "sparse_structure_decoder", "shape_slat_decoder",
                             "tex_slat_decoder", "shape_slat_flow_model_512", "tex_slat_flow_model_512"}  # nothing released


def test_1024_cascade_args_and_release(img):
    p = _Pipe("1024_cascade")
    log = []
    st = stages.sample_stages(p, img, "1024_cascade", steps={"ss": 6, "tex": 8}, max_num_tokens=30000,
                              release=True, log=log.append)
    kinds = [c[0] for c in p.calls]
    assert kinds == ["preprocess", "cond", "cond", "ss", "shape_cascade", "tex"]
    assert [c[1] for c in p.calls if c[0] == "cond"] == [512, 1024]
    ss = p.calls[3]; sc = p.calls[4]; tx = p.calls[5]
    assert ss[1] == 512 and ss[2] == 32 and ss[3] == {"steps": 6}
    assert sc[1:5] == (512, 1024, 512, 1024) and sc[5] == {} and sc[6] == 30000
    assert tx[1] == 1024 and tx[2] == {"steps": 8} and tx[3] == 300
    assert st["res"] == 1024
    # everything but the decoders was released, in stage order
    assert set(p.models) == {"shape_slat_decoder", "tex_slat_decoder"}
    assert p.image_cond_model is None and p.rembg_model is None
    assert any("sparse_structure_flow_model" in l for l in log)


def test_resume_skips_done_stages_and_restores_rng(img):
    p = _Pipe("512")
    st = stages.sample_stages(p, img, "512", log=lambda *_: None)
    partial = {k: v for k, v in st.items() if k != "tex_feats"}
    partial["done"] = ["coords", "shape"]
    p2 = _Pipe("512")
    st2 = stages.sample_stages(p2, img, "512", state=partial, log=lambda *_: None)
    kinds = [c[0] for c in p2.calls]
    assert kinds == ["cond", "tex"]                # no preprocess (image_png reused), no ss/shape
    assert st2["done"] == ["coords", "shape", "tex"]
    assert torch.equal(st2["coords"], st["coords"])
    with pytest.raises(ValueError, match="different"):
        stages.sample_stages(p2, img, "512", seed=99, state=partial, log=lambda *_: None)


def test_memory_pressure_stops_and_keeps_completed_stages(img):
    p = _Pipe("1024_cascade")
    p.fail_at = "shape"
    saved = []
    with pytest.raises(stages.MemoryPressure):
        stages.sample_stages(p, img, "1024_cascade", on_stage=lambda n, s: saved.append((n, dict(s))),
                             log=lambda *_: None)
    assert [n for n, _ in saved] == ["coords"]
    assert "coords" in saved[0][1] and saved[0][1]["done"] == ["coords"]


def test_guard_trips_between_steps(img):
    p = _Pipe("512")
    free = iter([10.0, 10.0, 1.0, 1.0, 1.0, 1.0])
    with pytest.raises(stages.MemoryPressure, match="floor"):
        stages.sample_stages(p, img, "512", min_free_gb=4.0, probe=lambda: next(free, 1.0),
                             log=lambda *_: None)
    # sample_once wrappers were removed again
    assert p.sparse_structure_sampler.sample_once.__name__ == "sample_once"


def test_guard_restores_sampler_methods_when_not_tripped(img):
    p = _Pipe("512")
    stages.sample_stages(p, img, "512", min_free_gb=1.0, probe=lambda: 50.0, log=lambda *_: None)
    assert p.shape_slat_sampler.sample_once.__name__ == "sample_once"


def test_decode_state(img):
    p = _Pipe("512")
    st = stages.sample_stages(p, img, "512", log=lambda *_: None)
    mesh = stages.decode_state(p, st, log=lambda *_: None)
    assert mesh.vertices.shape == (3, 3)
    assert p.calls[-1][0] == "decode" and p.calls[-1][1] == 512
    with pytest.raises(ValueError, match="incomplete"):
        stages.decode_state(p, {"done": ["coords"]})


def test_steps_for():
    assert stages._steps_for(None, "tex") == {}
    assert stages._steps_for(9, "shape") == {"steps": 9}
    assert stages._steps_for({"ss": 4, "all": 7}, "coords") == {"steps": 4}
    assert stages._steps_for({"ss": 4, "all": 7}, "tex") == {"steps": 7}
    assert stages._steps_for({"ss": 4}, "tex") == {}
