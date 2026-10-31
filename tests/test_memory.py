import sys
import types

import pytest

from trellis3 import memory


def test_models_for_each_type():
    assert memory.models_for(["512"]) == memory.ALWAYS + [
        "shape_slat_flow_model_512", "tex_slat_flow_model_512"]
    m1024 = memory.models_for(["1024_cascade"])
    assert "tex_slat_flow_model_512" not in m1024
    assert "shape_slat_flow_model_1024" in m1024
    assert memory.models_for(["1536_cascade"]) == m1024
    both = memory.models_for(["512", "1024_cascade"])
    assert len(both) == 8  # everything


def test_models_for_rejects_unknown():
    with pytest.raises(ValueError):
        memory.models_for(["2048"])


def test_savings_are_real():
    full = memory.approx_gb(memory.models_for(["512", "1024_cascade"]))
    only512 = memory.approx_gb(memory.models_for(["512"]))
    assert full - only512 > 5.0  # the two unused 1024 flow models
    only1024 = memory.approx_gb(memory.models_for(["1024_cascade"]))
    assert full - only1024 > 2.5  # the unused tex 512 flow model


class _FakePipeline:
    model_names_to_load = ["everything"]
    seen = None

    def __init__(self):
        self.models = {}
        self.low_vram = True
        self._pretrained_args = {"models": {n: f"ckpts/{n}" for n in memory.APPROX_BYTES}}
        self.device = "cpu"

    @classmethod
    def from_pretrained(cls, path):
        cls.seen = list(cls.model_names_to_load)
        p = cls()
        for n in cls.model_names_to_load:
            p.models[n] = _FakeModel(n)
        return p


class _FakeModel:
    def __init__(self, name):
        self.name = name
        self.dev = "cpu"
        self.evaled = False

    def eval(self):
        self.evaled = True

    def to(self, d):
        self.dev = d


@pytest.fixture
def fake_trellis2(monkeypatch):
    pipe_mod = types.ModuleType("trellis2.pipelines.trellis2_image_to_3d")
    pipe_mod.Trellis2ImageTo3DPipeline = _FakePipeline
    models_mod = types.ModuleType("trellis2.models")
    models_mod.from_pretrained = lambda path: _FakeModel(path)
    for name, mod in {
        "trellis2": types.ModuleType("trellis2"),
        "trellis2.pipelines": types.ModuleType("trellis2.pipelines"),
        "trellis2.pipelines.trellis2_image_to_3d": pipe_mod,
        "trellis2.models": models_mod,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)
    _FakePipeline.model_names_to_load = ["everything"]
    return pipe_mod


def test_load_pipeline_restricts_and_restores(fake_trellis2):
    p = memory.load_pipeline("512")
    assert set(p.models) == set(memory.models_for(["512"]))
    assert _FakePipeline.seen == memory.models_for(["512"])
    assert _FakePipeline.model_names_to_load == ["everything"]  # restored
    assert p.low_vram is True


def test_load_pipeline_resident(fake_trellis2):
    p = memory.load_pipeline(["512"], resident=True)
    assert p.low_vram is False


def test_ensure_models_lazy_loads(fake_trellis2):
    p = memory.load_pipeline("512")
    added = memory.ensure_models(p, "1024_cascade")
    assert set(added) == {"shape_slat_flow_model_1024", "tex_slat_flow_model_1024"}
    assert all(p.models[n].evaled for n in added)
    assert memory.ensure_models(p, "1024_cascade") == []


def test_ensure_models_moves_to_device_when_resident(fake_trellis2):
    p = memory.load_pipeline("512", resident=True)
    p.device = "mps"
    added = memory.ensure_models(p, "1024")
    assert p.models[added[0]].dev == "mps"


def test_stage_hooks_fire_after_each_stage():
    calls = []

    class P:
        def sample_sparse_structure(self, x):
            calls.append("ss")
            return x * 2

        def decode_tex_slat(self, a, b=1):
            calls.append("tex")
            return a + b

    p = P()
    fired = []
    memory.install_stage_hooks(p, after=lambda: fired.append(1))
    assert p.sample_sparse_structure(3) == 6
    assert p.decode_tex_slat(1, b=2) == 3
    assert calls == ["ss", "tex"]
    assert len(fired) == 2
    # idempotent
    memory.install_stage_hooks(p, after=lambda: fired.append(99))
    p.sample_sparse_structure(1)
    assert 99 not in fired


def test_free_device_cache_is_safe_on_cpu():
    memory.free_device_cache()


def test_rss_gb_returns_number_or_none():
    r = memory.rss_gb()
    assert r is None or r > 0


VM_STAT = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                               12345.
Pages active:                            500000.
Pages inactive:                          100000.
Pages speculative:                         5000.
Pages throttled:                              0.
"""


def test_parse_vm_stat():
    gb = memory.parse_vm_stat(VM_STAT)
    assert abs(gb - (12345 + 100000 + 5000) * 16384 / 1e9) < 1e-6
    assert memory.parse_vm_stat("") is None


def test_parse_meminfo():
    assert memory.parse_meminfo("MemTotal: 10 kB\nMemAvailable:  4000000 kB\n") == 4.0
    assert memory.parse_meminfo("nothing") is None


def test_free_memory_gb_runs():
    v = memory.free_memory_gb()
    assert v is None or v > 0


def test_load_pipeline_decode_only_stubs_cond_models(fake_trellis2, monkeypatch, tmp_path):
    import json
    ife = types.ModuleType("trellis2.modules.image_feature_extractor")
    class Dino:
        def __init__(self, **k): raise AssertionError("real DINOv3 must not be built")
    ife.DinoV3FeatureExtractor = Dino
    rem = types.ModuleType("trellis2.pipelines.rembg")
    class Bi:
        def __init__(self, **k): raise AssertionError("real BiRefNet must not be built")
    rem.BiRefNet = Bi
    monkeypatch.setitem(sys.modules, "trellis2.modules", types.ModuleType("trellis2.modules"))
    monkeypatch.setitem(sys.modules, "trellis2.modules.image_feature_extractor", ife)
    monkeypatch.setitem(sys.modules, "trellis2.pipelines.rembg", rem)
    (tmp_path / "pipeline.json").write_text(json.dumps({"args": {
        "models": {}, "image_cond_model": {"name": "DinoV3FeatureExtractor", "args": {}},
        "rembg_model": {"name": "BiRefNet", "args": {}}}}))

    # the fake from_pretrained does not build cond models, so emulate upstream here
    orig = _FakePipeline.from_pretrained.__func__
    def from_pretrained(cls, path):
        p = orig(cls, path)
        p.image_cond_model = ife.DinoV3FeatureExtractor(model_name="x")
        p.rembg_model = rem.BiRefNet(model_name="y")
        return p
    monkeypatch.setattr(_FakePipeline, "from_pretrained", classmethod(from_pretrained))

    p = memory.load_pipeline("1024_cascade", path=str(tmp_path), decode_only=True)
    assert set(p.models) == set(memory.DECODE_ONLY)
    assert p.image_cond_model is None and p.rembg_model is None
    assert ife.DinoV3FeatureExtractor is Dino and rem.BiRefNet is Bi      # restored
    assert _FakePipeline.model_names_to_load == ["everything"]


def test_pick_device_prefers_and_validates(monkeypatch):
    from trellis3 import runner
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert runner.pick_device("cpu").type == "cpu"
    assert runner.pick_device("auto").type in ("cpu", "mps")
    with pytest.raises(RuntimeError, match="ROCm"):
        runner.pick_device("rocm")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert runner.pick_device("rocm").type == "cuda"
    assert runner.pick_device("cuda").type == "cuda"
    monkeypatch.setenv("TRELLIS3_DEVICE", "cpu")
    assert runner.pick_device().type == "cpu"
    assert runner.describe_device(torch.device("cpu")) == "cpu"
