"""The vectorised mesh extraction must equal the upstream dict version."""
import torch
import trimesh

from backends import mesh_extract as me


def _reference_quads(coords, intersected_flag, offsets):
    """Upstream's dict-based neighbour lookup, kept here as the oracle."""
    N = coords.shape[0]
    d = {tuple(c): i for i, c in enumerate(coords.tolist())}
    edge_neighbor = coords.reshape(N, 1, 1, 3) + offsets
    connected = edge_neighbor[intersected_flag].reshape(-1, 3)
    idx = [d.get(tuple(c), 0xFFFFFFFF) for c in connected.tolist()]
    idx = torch.tensor(idx, dtype=torch.int64).reshape(-1, 4)
    valid = (idx != 0xFFFFFFFF).all(dim=1)
    return idx[valid]


def _shell(res=24, r=0.4):
    """Voxels within one cell of a sphere surface — a plausible decoder grid."""
    g = torch.arange(res)
    zz, yy, xx = torch.meshgrid(g, g, g, indexing="ij")
    c = torch.stack([zz, yy, xx], dim=-1).reshape(-1, 3)
    p = (c.float() + 0.5) / res - 0.5
    dist = p.norm(dim=1)
    keep = (dist - r).abs() < 1.5 / res
    return c[keep]


def test_matches_reference_and_builds_a_mesh():
    coords = _shell()
    N = len(coords)
    gen = torch.Generator().manual_seed(0)
    dual = torch.rand((N, 3), generator=gen) * 0.6 + 0.2
    flags = torch.rand((N, 3), generator=gen) < 0.5

    v, f = me.flexible_dual_grid_to_mesh(coords, dual, flags, None,
                                         aabb=[[-0.5] * 3, [0.5] * 3], grid_size=24)
    # oracle on the quad index set (before triangulation, which is shared code)
    ref = _reference_quads(coords, flags, me._edge_neighbor_voxel_offset.squeeze(0))
    assert f.shape[0] == 2 * ref.shape[0]
    assert v.shape[0] == N
    tri = f.reshape(-1, 6)
    for col in range(6):
        assert ((tri[:, col][:, None] == ref).any(dim=1)).all()
    m = trimesh.Trimesh(v.numpy(), f.numpy(), process=False)
    assert len(m.faces) > 100


def test_no_intersections_returns_empty():
    coords = _shell(12)
    v, f = me.flexible_dual_grid_to_mesh(coords, torch.zeros((len(coords), 3)),
                                         torch.zeros((len(coords), 3), dtype=torch.bool), None,
                                         aabb=[[-0.5] * 3, [0.5] * 3], grid_size=12)
    assert v.shape == (0, 3) and f.shape == (0, 3)


def test_split_weight_path():
    coords = _shell(16)
    N = len(coords)
    gen = torch.Generator().manual_seed(1)
    dual = torch.rand((N, 3), generator=gen)
    flags = torch.rand((N, 3), generator=gen) < 0.6
    sw = torch.rand((N, 1), generator=gen)
    v, f = me.flexible_dual_grid_to_mesh(coords, dual, flags, sw,
                                         aabb=[[-0.5] * 3, [0.5] * 3], voxel_size=1 / 16)
    assert f.shape[1] == 3 and f.max() < N
