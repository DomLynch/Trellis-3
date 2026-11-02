"""Export option (a): with the GPU baker the raw mesh stays on the device and to_glb decimates it there (no host copy, no CPU simplify); the OBJ dump is optional. CPU only, to_glb stubbed."""
import os
import sys
import types

import numpy as np
import pytest
import torch
import trimesh
from PIL import Image

import trellis3.export as ex
from trellis3.export import export_mesh


class _Mesh:
    def __init__(self, v, f):
        self.vertices = torch.tensor(v, dtype=torch.float32)
        self.faces = torch.tensor(f, dtype=torch.int32)
        self.attrs = torch.zeros((1, 6))
        self.coords = torch.zeros((1, 3), dtype=torch.int32)
        self.layout = {}
        self.voxel_size = 1 / 512
        self.host_copies = 0


@pytest.fixture
def stub(monkeypatch):
    seen = {}

    def to_glb(**kw):
        seen.update(kw)
        m = trimesh.Trimesh(kw["vertices"].numpy(), kw["faces"].numpy(), process=False)
        m.visual = trimesh.visual.TextureVisuals(uv=np.zeros((len(m.vertices), 2)), material=trimesh.visual.material.PBRMaterial(
            baseColorTexture=Image.fromarray(np.full((2, 2, 3), 64, np.uint8)), metallicFactor=1.0))
        return m

    ov = types.ModuleType("o_voxel")   # the real package initialises the GPU on import
    ov.postprocess = types.ModuleType("o_voxel.postprocess")
    ov.postprocess.to_glb = to_glb
    monkeypatch.setitem(sys.modules, "o_voxel", ov)
    monkeypatch.setitem(sys.modules, "o_voxel.postprocess", ov.postprocess)
    import trellis3.glb_lean as gl
    monkeypatch.setattr(gl, "get_to_glb", lambda: (to_glb, True))
    monkeypatch.setattr(ex, "gpu_bake_backend", lambda: "cuda")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)   # bake_dev = cpu
    monkeypatch.delenv("TRELLIS3_GPU_SIMPLIFY", raising=False)
    monkeypatch.delenv("TRELLIS3_OBJ", raising=False)
    return seen


def _mesh():
    ico = trimesh.creation.icosphere(subdivisions=3)   # 1280 faces
    m = _Mesh(np.asarray(ico.vertices), np.asarray(ico.faces))
    return m, len(ico.faces)


def _spy_host(m, monkeypatch):
    real = ex.torch.Tensor.cpu

    def spy(self, *a, **k):
        if self is m.vertices or self is m.faces:
            m.host_copies += 1
        return real(self, *a, **k)
    monkeypatch.setattr(ex.torch.Tensor, "cpu", spy)


def test_gpu_simplify_hands_the_raw_mesh_to_to_glb_and_never_copies_it_to_the_host(tmp_path, stub, monkeypatch):
    m, n = _mesh()
    _spy_host(m, monkeypatch)
    out = export_mesh(m, str(tmp_path / "g"), texture_size=64, bake_faces=100, gpu_simplify=True, write_obj=False, log=lambda *_: None)
    assert out["gpu_simplify"] is True and out["faces"] == n
    assert stub["faces"].shape[0] == n                 # to_glb got the RAW mesh: it is the one that decimates (cumesh), to decimation_target
    assert stub["decimation_target"] == 100
    assert m.host_copies == 0                          # no .cpu().numpy() of the raw mesh
    assert not (tmp_path / "g.obj").exists()


def test_default_still_simplifies_on_the_host_and_the_env_switch_turns_the_gpu_path_on(tmp_path, stub, monkeypatch):
    m, n = _mesh()
    out = export_mesh(m, str(tmp_path / "h"), texture_size=64, bake_faces=100, log=lambda *_: None)
    assert out["gpu_simplify"] is False and stub["faces"].shape[0] < n
    monkeypatch.setenv("TRELLIS3_GPU_SIMPLIFY", "1")
    m2, _ = _mesh()
    out = export_mesh(m2, str(tmp_path / "e"), texture_size=64, bake_faces=100, log=lambda *_: None)
    assert out["gpu_simplify"] is True and stub["faces"].shape[0] == n


def test_obj_dump_is_automatic_only_for_small_meshes(tmp_path, stub, monkeypatch):
    m, n = _mesh()
    export_mesh(m, str(tmp_path / "s"), texture_size=64, bake_faces=100, log=lambda *_: None)
    assert (tmp_path / "s.obj").exists()                # small mesh: written as before
    monkeypatch.setattr(ex, "_simplify", lambda v, f, t, log: (v, f))
    m2, _ = _mesh()
    big = types.SimpleNamespace(shape=(6_000_000, 3))
    m2.faces = types.SimpleNamespace(shape=big.shape, cpu=lambda: None)   # pretend a 6M-face mesh: the rule only reads the count
    with pytest.raises(Exception):
        export_mesh(m2, str(tmp_path / "b"), texture_size=64, bake_faces=100, gpu_simplify=True, log=lambda *_: None)
    assert not (tmp_path / "b.obj").exists()
