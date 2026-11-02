import sys
import types

import pytest
import torch
from PIL import Image

from trellis3 import stages


class _Sparse:
    def __init__(self, feats, coords):
        self.feats, self.coords = feats, coords

    @property
    def device(self):
        return self.feats.device

    def replace(self, feats):
        return _Sparse(feats, self.coords)

    def __mul__(self, other):
        return _Sparse(self.feats * other, self.coords)

    def __add__(self, other):
        return _Sparse(self.feats + other, self.coords)


class _Sampler:
    def __init__(self):
        self.calls = 0

    def sample_once(self, *a, **k):
        self.calls += 1
        return None


class _Model:
    in_channels = 32

    def __init__(self, name):
        self.name = name
        self.where = "cpu"
        self.low_vram = False

    def to(self, d):
        self.where = str(d)

    def cpu(self):
        self.where = "cpu"

    def upsample(self, slat, upsample_times):
        # 16x finer coords than the input, batch column preserved
        base = slat.coords.repeat_interleave(4, dim=0).clone()
        base[:, 1:] = base[:, 1:] * 16 + torch.randint(0, 16, (base.shape[0], 3), dtype=base.dtype)
        return base


class _ShapeSampler(_Sampler):
    def sample(self, model, noise, cond=None, steps=None, verbose=False, tqdm_desc="", **kw):
        self.sample_once()          # one "step", so the memory guard sees it
        self.passes = getattr(self, "passes", 0) + 1
        self.last = {"model": model.name, "cond": cond, "steps": steps, "n": noise.feats.shape[0], "desc": tqdm_desc}
        return types.SimpleNamespace(samples=_Sparse(torch.randn(noise.feats.shape[0], 32), noise.coords))


class _Pipe:
    """Records every stage call with its arguments."""

    def __init__(self, ptype):
        from trellis3 import memory
        self.models = {n: _Model(n) for n in memory.models_for([ptype])}
        self.device = "cpu"
        self.low_vram = True
        self.shape_slat_sampler_params = {"steps": 12, "guidance_strength": 7.5}
        self.shape_slat_normalization = {"mean": [0.0] * 32, "std": [1.0] * 32}
        self.image_cond_model = object()
        self.rembg_model = object()
        self.sparse_structure_sampler = _Sampler()
        self.shape_slat_sampler = _ShapeSampler()
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
        c = torch.randint(0, 32, (100, 4), dtype=torch.int32)
        c[:, 0] = 0
        return c

    def sample_shape_slat(self, cond, model, coords, params):
        self.calls.append(("shape", cond["cond"], params))
        self._step(self.shape_slat_sampler)
        return _Sparse(torch.randn(100, 32), coords)

    def sample_shape_slat_cascade(self, *a, **k):
        raise AssertionError("upstream cascade must not be called; Trellis-3 splits it")

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


def test_1024_cascade_split_into_three_saved_substages(img):
    p = _Pipe("1024_cascade")
    seen = []
    log = []
    st = stages.sample_stages(p, img, "1024_cascade", steps={"ss": 6, "shape": 5, "tex": 8}, max_num_tokens=30000,
                              release=True, on_stage=lambda n, s: seen.append((n, sorted(k for k in s if k.endswith("feats") or k.endswith("coords")))),
                              log=log.append)
    kinds = [c[0] for c in p.calls]
    assert kinds == ["preprocess", "cond", "cond", "ss", "tex"]        # shape ran through the split path
    assert [n for n, _ in seen] == ["coords", "shape_lr", "shape_up", "shape_hr", "shape", "tex"]
    # 512 latents are on disk before the upsample, hr coords before the 1024 pass
    assert "shape_lr_feats" in seen[1][1] and "shape_hr_coords" not in seen[1][1]
    assert "shape_hr_coords" in seen[2][1] and "shape_feats" not in seen[2][1]
    assert "shape_feats" in seen[3][1]
    # the two sampling passes used the right model/cond/steps and 16x finer coords
    assert p.shape_slat_sampler.passes == 2
    last = p.shape_slat_sampler.last
    assert last["model"] == "shape_slat_flow_model_1024" and last["cond"] == 1024 and last["steps"] == 5
    assert last["n"] == st["shape_hr_coords"].shape[0] and st["res"] == 1024
    assert st["shape_hr_coords"][:, 1:].max() < 64                   # quantised to res//16
    # release order: 512 flow dropped right after shape_lr, 1024 after shape_hr
    i_lr = next(i for i, l in enumerate(log) if "shape_slat_flow_model_512" in l)
    i_up = next(i for i, l in enumerate(log) if "stage shape_up" in l)
    assert i_lr < i_up
    assert set(p.models) == {"shape_slat_decoder", "tex_slat_decoder"}
    assert p.models["shape_slat_decoder"].where == "cpu" and p.models["shape_slat_decoder"].low_vram is False


def test_1024_cascade_resumes_after_lr(img):
    p = _Pipe("1024_cascade")
    st = stages.sample_stages(p, img, "1024_cascade", log=lambda *_: None)
    partial = {k: v for k, v in st.items() if k not in ("shape_hr_coords", "shape_feats", "shape_coords", "tex_feats", "res")}
    partial["done"] = ["coords", "shape_lr"]
    p2 = _Pipe("1024_cascade")
    st2 = stages.sample_stages(p2, img, "1024_cascade", state=partial, log=lambda *_: None)
    assert p2.shape_slat_sampler.passes == 1                     # only the 1024 pass re-ran
    assert p2.shape_slat_sampler.last["cond"] == 1024
    assert st2["done"] == ["coords", "shape_lr", "shape_up", "shape_hr", "shape", "tex"]
    assert torch.equal(st2["shape_lr_feats"], st["shape_lr_feats"])


def test_token_budget_reduces_resolution(img):
    p = _Pipe("1024_cascade")
    st = stages.sample_stages(p, img, "1024_cascade", max_num_tokens=10, log=lambda *_: None)
    assert st["res"] == 1024                                     # floor: never below 1024
    p = _Pipe("1536_cascade")
    st = stages.sample_stages(p, img, "1536_cascade", max_num_tokens=10, log=lambda *_: None)
    assert st["res"] == 1024                                     # stepped down from 1536 by 128s


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
    saved = []
    free = iter([50.0, 50.0, 0.5])       # ss step ok, 512 pass ok, 1024 pass trips
    with pytest.raises(stages.MemoryPressure):
        stages.sample_stages(p, img, "1024_cascade", min_free_gb=2.0, probe=lambda: next(free, 0.5),
                             on_stage=lambda n, s: saved.append((n, dict(s))), log=lambda *_: None)
    assert [n for n, _ in saved] == ["coords", "shape_lr", "shape_up"]
    assert "shape_hr_coords" in saved[-1][1] and saved[-1][1]["done"] == ["coords", "shape_lr", "shape_up"]


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


def test_lean_decode_order_and_output(monkeypatch):
    """decode_shape -> decode_tex -> (release) -> fill_holes, and the same MeshWithVoxel fields as upstream."""
    order = []

    class Mesh:
        vertices = torch.zeros((3, 3))
        faces = torch.tensor([[0, 1, 2]])

        def fill_holes(self):
            order.append("fill_holes")

    class Vox:
        coords = torch.zeros((2, 4), dtype=torch.int32)
        feats = torch.zeros((2, 6))
        shape = torch.Size([2, 6])
        spatial_shape = torch.Size([4, 4, 4])

    class MWV:
        def __init__(self, vertices, faces, **kw):
            self.vertices, self.faces, self.kw = vertices, faces, kw

    reps = types.ModuleType("trellis2.representations")
    reps.MeshWithVoxel = MWV
    monkeypatch.setitem(sys.modules, "trellis2.representations", reps)

    class P:
        pbr_attr_layout = {"base_color": slice(0, 3)}

        def decode_shape_slat(self, slat, res):
            order.append("shape")
            return [Mesh()], ["subs"]

        def decode_tex_slat(self, slat, subs):
            order.append("tex")
            assert subs == ["subs"]
            return [Vox()]

    out = stages.decode_latent_lean(P(), object(), object(), 1024)
    assert order == ["shape", "tex", "fill_holes"]
    assert out[0].kw["voxel_size"] == 1 / 1024 and out[0].kw["origin"] == [-0.5, -0.5, -0.5]
    assert out[0].kw["coords"].shape == (2, 3)
