"""
Apply the mesh-op patch to a copy of upstream base.py and exercise the
patched class for real (inside a throwaway package with stubbed deps).
"""
import importlib
import os
import shutil
import sys
import textwrap

import numpy as np
import pytest
import torch
import trimesh

import patches.mps_compat as pc

UPSTREAM_STUB = "        return  # Skip — Metal cumesh segfaults on large decode meshes\n"


def _setup_root(tmp_path, trellis2_src):
    root = tmp_path / "TRELLIS.2"
    mesh_dir = root / "trellis2/representations/mesh"
    mesh_dir.mkdir(parents=True)
    shutil.copy(os.path.join(trellis2_src, "trellis2/representations/mesh/base.py"), mesh_dir / "base.py")
    return root


def _apply(monkeypatch, root):
    monkeypatch.setattr(pc, "TRELLIS_ROOT", str(root))
    pc.patch_mesh_base()
    pc.install_mesh_cleanup()
    pc.patch_mesh_base_cpu_ops()
    return (root / "trellis2/representations/mesh/base.py").read_text()


def _import_patched(tmp_path, root, name):
    """Build a fake package around the patched base.py and import Mesh."""
    pkg = tmp_path / name
    (pkg / "mesh").mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "voxel.py").write_text("class Voxel:\n    pass\n")
    (pkg / "mesh" / "__init__.py").write_text("")
    for fn in ("base.py", "cleanup_cpu.py"):
        shutil.copy(root / "trellis2/representations/mesh" / fn, pkg / "mesh" / fn)
    stubs = tmp_path / "stubs"
    (stubs / "flex_gemm" / "ops").mkdir(parents=True)
    (stubs / "cumesh.py").write_text("")
    (stubs / "flex_gemm" / "__init__.py").write_text("")
    (stubs / "flex_gemm" / "ops" / "__init__.py").write_text("")
    (stubs / "flex_gemm" / "ops" / "grid_sample.py").write_text("def grid_sample_3d(*a, **k): raise RuntimeError\n")
    sys.path.insert(0, str(stubs))
    sys.path.insert(0, str(tmp_path))
    try:
        for m in list(sys.modules):
            if m == name or m.startswith(name + "."):
                del sys.modules[m]
        return importlib.import_module(f"{name}.mesh.base").Mesh
    finally:
        sys.path.remove(str(stubs))
        sys.path.remove(str(tmp_path))


def test_patch_fresh_upstream_file(tmp_path, monkeypatch, trellis2_src):
    root = _setup_root(tmp_path, trellis2_src)
    src = _apply(monkeypatch, root)
    assert "cumesh.CuMesh()" not in src
    assert src.count("from .cleanup_cpu import") == 3
    assert "class TextureFilterMode" in src  # following classes intact
    assert "class MeshWithVoxel" in src
    compile(src, "base.py", "exec")


def test_patch_already_stubbed_by_upstream(tmp_path, monkeypatch, trellis2_src):
    """Existing trellis-mac installs already have the `return` stubs."""
    root = _setup_root(tmp_path, trellis2_src)
    monkeypatch.setattr(pc, "TRELLIS_ROOT", str(root))
    pc.patch_mesh_base()
    src = (root / "trellis2/representations/mesh/base.py").read_text()
    assert UPSTREAM_STUB in src
    pc.install_mesh_cleanup()
    pc.patch_mesh_base_cpu_ops()
    src = (root / "trellis2/representations/mesh/base.py").read_text()
    assert UPSTREAM_STUB not in src
    assert src.count("from .cleanup_cpu import") == 3


def test_patch_is_idempotent(tmp_path, monkeypatch, trellis2_src):
    root = _setup_root(tmp_path, trellis2_src)
    once = _apply(monkeypatch, root)
    twice = _apply(monkeypatch, root)
    assert once == twice


def test_patched_mesh_fills_hole_for_real(tmp_path, monkeypatch, trellis2_src):
    root = _setup_root(tmp_path, trellis2_src)
    _apply(monkeypatch, root)
    Mesh = _import_patched(tmp_path, root, "fakepkg")

    ico = trimesh.creation.icosphere(subdivisions=2)
    faces = np.asarray(ico.faces)[1:]
    m = Mesh(torch.tensor(np.asarray(ico.vertices), dtype=torch.float32),
             torch.tensor(faces, dtype=torch.int32))
    nf = m.faces.shape[0]
    m.fill_holes(max_hole_perimeter=10.0)
    assert m.faces.shape[0] == nf + 3
    assert m.faces.dtype == torch.int32
    assert trimesh.Trimesh(m.vertices.numpy(), m.faces.numpy(), process=False).is_watertight

    m.remove_faces(torch.zeros(m.faces.shape[0], dtype=torch.bool))
    assert m.faces.shape[0] == nf + 3

    m.simplify(target=40)
    assert m.faces.shape[0] < nf


def test_patch_fails_loudly_if_method_missing(tmp_path, monkeypatch):
    root = tmp_path / "TRELLIS.2"
    mesh_dir = root / "trellis2/representations/mesh"
    mesh_dir.mkdir(parents=True)
    (mesh_dir / "base.py").write_text(textwrap.dedent("""
        class Mesh:
            def fill_holes(self, max_hole_perimeter=3e-2):
                pass
    """))
    monkeypatch.setattr(pc, "TRELLIS_ROOT", str(root))
    with pytest.raises(RuntimeError, match="remove_faces"):
        pc.patch_mesh_base_cpu_ops()
