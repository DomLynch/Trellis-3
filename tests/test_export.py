import numpy as np
from PIL import Image
import pytest
import torch
import trimesh

from trellis3.export import export_mesh, output_stem


class _Mesh:
    def __init__(self, v, f, attrs=None):
        self.vertices = torch.tensor(v, dtype=torch.float32)
        self.faces = torch.tensor(f, dtype=torch.int32)
        self.attrs = attrs


def test_export_vertex_colour_path(tmp_path):
    box = trimesh.creation.box()
    m = _Mesh(np.asarray(box.vertices), np.asarray(box.faces))
    out = export_mesh(m, str(tmp_path / "box"), no_texture=True, log=lambda *_: None)
    assert out["bake"] == "none"
    assert (tmp_path / "box.glb").exists() and (tmp_path / "box.obj").exists()
    assert out["faces"] == 12 and out["vertices"] == 8
    re = trimesh.load(str(tmp_path / "box.obj"), force="mesh")
    assert len(re.faces) == 12


def test_export_without_voxels_falls_back_even_if_texture_requested(tmp_path):
    box = trimesh.creation.box()
    m = _Mesh(np.asarray(box.vertices), np.asarray(box.faces), attrs=None)
    out = export_mesh(m, str(tmp_path / "b"), texture_size=1024, no_texture=False,
                      log=lambda *_: None)
    assert out["bake"] == "none"


def test_export_empty_mesh_raises(tmp_path):
    m = _Mesh(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64))
    with pytest.raises(RuntimeError, match="empty"):
        export_mesh(m, str(tmp_path / "e"), no_texture=True, log=lambda *_: None)


def test_output_stem(tmp_path):
    s = output_stem("seed3", str(tmp_path / "jobs" / "img"))
    assert s.endswith("img/seed3") and (tmp_path / "jobs" / "img").is_dir()
    assert output_stem("plain") == "plain"


def test_metal_path_wiring(tmp_path, monkeypatch):
    """Fake o_voxel.to_glb: check winding fix, uv kwargs, metallic and gamma reach it."""
    import sys, types
    import trellis3.export as ex

    seen = {}
    icos = trimesh.creation.icosphere(subdivisions=2)
    faces = np.asarray(icos.faces).copy()
    faces[: len(faces) // 2] = faces[: len(faces) // 2, ::-1]  # half flipped

    def to_glb(**kw):
        seen.update(kw)
        m = trimesh.Trimesh(kw["vertices"].numpy(), kw["faces"].numpy(), process=False)
        seen["winding_ok"] = m.is_winding_consistent
        tex = Image.fromarray(np.full((2, 2, 3), 64, np.uint8))
        m.visual = trimesh.visual.TextureVisuals(
            uv=np.zeros((len(m.vertices), 2)),
            material=trimesh.visual.material.PBRMaterial(baseColorTexture=tex, metallicFactor=1.0))
        seen["glb"] = m
        return m

    pp = types.ModuleType("o_voxel.postprocess"); pp.to_glb = to_glb
    ov = types.ModuleType("o_voxel"); ov.postprocess = pp
    monkeypatch.setitem(sys.modules, "o_voxel", ov)
    monkeypatch.setitem(sys.modules, "o_voxel.postprocess", pp)
    monkeypatch.setattr(ex, "_metal_bake_available", lambda: True)

    m = _Mesh(np.asarray(icos.vertices), faces)
    m.attrs = torch.zeros((1, 6)); m.coords = torch.zeros((1, 3), dtype=torch.int32)
    m.layout = {}; m.voxel_size = 1 / 512
    out = export_mesh(m, str(tmp_path / "x"), texture_size=512, metallic_factor=0.0, gamma=2.2,
                      uv_cone_deg=120, uv_smooth=2, log=lambda *_: None)
    assert out["bake"] == "metal" and (tmp_path / "x.glb").exists()
    assert seen["winding_ok"] is True
    assert abs(seen["mesh_cluster_threshold_cone_half_angle_rad"] - np.radians(120)) < 1e-9
    assert seen["mesh_cluster_smooth_strength"] == 2
    assert seen["texture_size"] == 512
    assert seen["glb"].visual.material.metallicFactor == 0.0
    assert np.asarray(seen["glb"].visual.material.baseColorTexture)[0, 0, 0] > 100


def test_metal_path_normal_map(tmp_path, monkeypatch):
    import sys, types
    import trellis3.export as ex

    icos = trimesh.creation.icosphere(subdivisions=3, radius=0.5)
    seen = {}

    def to_glb(**kw):
        # Mimic to_glb: unwrap (planar uv) and convert Y-up -> Z-up
        v = kw["vertices"].numpy().copy()
        f = kw["faces"].numpy()
        uv = np.stack([np.arctan2(v[:, 1], v[:, 0]) / (2 * np.pi) + 0.5,
                       np.arccos(np.clip(v[:, 2] / 0.5, -1, 1)) / np.pi], axis=1)
        keep = np.ptp(uv[f][:, :, 0], axis=1) < 0.5
        g = v.copy(); g[:, 1], g[:, 2] = v[:, 2].copy(), -v[:, 1].copy()
        m = trimesh.Trimesh(g, f[keep], process=False)
        m.visual = trimesh.visual.TextureVisuals(
            uv=uv, material=trimesh.visual.material.PBRMaterial(
                baseColorTexture=Image.fromarray(np.zeros((2, 2, 3), np.uint8))))
        seen["glb"] = m
        return m

    pp = types.ModuleType("o_voxel.postprocess"); pp.to_glb = to_glb
    ov = types.ModuleType("o_voxel"); ov.postprocess = pp
    monkeypatch.setitem(sys.modules, "o_voxel", ov)
    monkeypatch.setitem(sys.modules, "o_voxel.postprocess", pp)
    monkeypatch.setattr(ex, "_metal_bake_available", lambda: True)
    monkeypatch.setattr(ex, "_simplify", lambda v, f, t, log: (v, f))  # keep full mesh

    m = _Mesh(np.asarray(icos.vertices), np.asarray(icos.faces))
    m.attrs = torch.zeros((1, 6)); m.coords = torch.zeros((1, 3), dtype=torch.int32)
    m.layout = {}; m.voxel_size = 1 / 512
    out = export_mesh(m, str(tmp_path / "n"), texture_size=64, normal_map=True,
                      fix_winding=False, log=lambda *_: None)
    assert out["normal"].endswith("n_normal.png") and (tmp_path / "n_normal.png").exists()
    tex = seen["glb"].visual.material.normalTexture
    assert tex is not None and tex.size == (64, 64)
    px = np.asarray(tex).reshape(-1, 3).astype(float)
    assert px[:, 2].mean() > 200   # same sphere hi/lo -> mostly flat
