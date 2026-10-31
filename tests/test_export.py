import numpy as np
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
