import numpy as np
import torch
import trimesh

from backends import mesh_cleanup as mc


def _tm(v, f):
    return trimesh.Trimesh(vertices=v, faces=f, process=False)


def test_closed_mesh_has_no_loops():
    m = trimesh.creation.icosphere(subdivisions=1)
    assert mc.boundary_loops(np.asarray(m.faces)) == []


def test_single_hole_detected(holey_mesh):
    v, f = holey_mesh
    loops = mc.boundary_loops(f)
    assert len(loops) == 1
    assert len(loops[0]) == 3


def test_fill_small_hole_makes_watertight(holey_mesh):
    v, f = holey_mesh
    assert not _tm(v, f).is_watertight
    v2, f2, n = mc.fill_holes_np(v, f, max_hole_perimeter=10.0)
    assert n == 1
    assert len(v2) == len(v) + 1
    assert len(f2) == len(f) + 3
    m = _tm(v2, f2)
    assert m.is_watertight
    assert m.is_winding_consistent
    assert m.volume > 0  # outward normals preserved


def test_large_hole_left_open(holey_mesh):
    v, f = holey_mesh
    # perimeter of one icosphere triangle at subdiv 2 is ~0.9; threshold below it
    v2, f2, n = mc.fill_holes_np(v, f, max_hole_perimeter=1e-3)
    assert n == 0
    assert len(f2) == len(f)


def test_multiple_holes_filled():
    m = trimesh.creation.icosphere(subdivisions=3)
    faces = np.asarray(m.faces)
    # remove two far-apart faces
    keep = np.ones(len(faces), bool)
    keep[0] = False
    keep[len(faces) // 2] = False
    v, f, n = mc.fill_holes_np(np.asarray(m.vertices), faces[keep], max_hole_perimeter=10.0)
    assert n == 2
    assert _tm(v, f).is_watertight


def test_remove_faces_compacts_vertices():
    m = trimesh.creation.box()
    v = np.asarray(m.vertices)
    f = np.asarray(m.faces)
    mask = np.zeros(len(f), bool)
    mask[:2] = True  # remove the first two faces (one side of the box)
    v2, f2 = mc.remove_faces_np(v, f, mask)
    assert len(f2) == len(f) - 2
    assert f2.max() < len(v2)
    assert len(v2) == len(np.unique(f[~mask]))


def test_remove_all_faces():
    m = trimesh.creation.box()
    v2, f2 = mc.remove_faces_np(np.asarray(m.vertices), np.asarray(m.faces),
                                np.ones(len(m.faces), bool))
    assert len(v2) == 0 and f2.shape == (0, 3)


def test_simplify_reduces_faces():
    m = trimesh.creation.icosphere(subdivisions=4)  # 5120 faces
    v, f = mc.simplify_np(np.asarray(m.vertices), np.asarray(m.faces), target=1000)
    assert 500 <= len(f) <= 1500
    assert len(f) < len(m.faces)


def test_simplify_noop_when_small():
    m = trimesh.creation.box()
    v, f = mc.simplify_np(np.asarray(m.vertices), np.asarray(m.faces), target=100)
    assert len(f) == len(m.faces)


def test_tensor_wrappers_preserve_dtype_and_device(holey_mesh):
    v, f = holey_mesh
    vt = torch.from_numpy(v).float()
    ft = torch.from_numpy(f.astype(np.int32))
    v2, f2 = mc.fill_holes_tensor(vt, ft, 10.0)
    assert v2.dtype == torch.float32 and f2.dtype == torch.int32
    assert v2.device == vt.device
    assert v2.shape[0] == vt.shape[0] + 1

    v3, f3 = mc.remove_faces_tensor(v2, f2, torch.zeros(f2.shape[0], dtype=torch.bool))
    assert f3.shape == f2.shape

    v4, f4 = mc.simplify_tensor(v2, f2, target=50)
    assert f4.shape[0] < f2.shape[0]


def test_fill_holes_tensor_noop_returns_same_objects():
    m = trimesh.creation.box()
    vt = torch.tensor(np.asarray(m.vertices), dtype=torch.float32)
    ft = torch.tensor(np.asarray(m.faces), dtype=torch.int32)
    v2, f2 = mc.fill_holes_tensor(vt, ft)
    assert v2 is vt and f2 is ft


def test_default_threshold_matches_cumesh_scale():
    """Meshes are in [-0.5,0.5]^3; a 3-edge hole of ~1e-2 edges must close at the default 3e-2."""
    m = trimesh.creation.icosphere(subdivisions=5, radius=0.5)  # edge ~0.03
    faces = np.asarray(m.faces)[1:]
    v, f, n = mc.fill_holes_np(np.asarray(m.vertices), faces)  # default threshold
    # edge length at subdiv 5 radius 0.5 is ~0.027, perimeter ~0.08 > 3e-2 -> stays open,
    # which is the intended conservative behaviour.
    assert n == 0
    m2 = trimesh.creation.icosphere(subdivisions=6, radius=0.5)  # edge ~0.014, perim ~0.04
    faces2 = np.asarray(m2.faces)[1:]
    v, f, n = mc.fill_holes_np(np.asarray(m2.vertices), faces2, max_hole_perimeter=0.05)
    assert n == 1
