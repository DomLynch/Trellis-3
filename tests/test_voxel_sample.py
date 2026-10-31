import numpy as np
import torch
import torch.nn.functional as F

from backends.voxel_sample import sparse_trilinear


def _dense_reference(vol, origin, vs, points):
    """torch grid_sample on the dense volume, align_corners=True — the official maths."""
    D, H, W, C = vol.shape
    g = (torch.tensor(points) - torch.tensor(origin)) / vs              # [M,3] index space (z,y,x order = dims)
    norm = torch.stack([g[:, 2] / (W - 1) * 2 - 1, g[:, 1] / (H - 1) * 2 - 1, g[:, 0] / (D - 1) * 2 - 1], dim=-1)
    out = F.grid_sample(torch.tensor(vol).permute(3, 0, 1, 2)[None], norm.reshape(1, 1, 1, -1, 3),
                        mode="bilinear", align_corners=True, padding_mode="zeros")
    return out[0, :, 0, 0, :].T.numpy()


def test_matches_official_grid_sample_on_full_grid():
    rng = np.random.default_rng(0)
    D = 9
    vol = rng.random((D, D, D, 6)).astype(np.float32)
    coords = np.stack(np.meshgrid(np.arange(D), np.arange(D), np.arange(D), indexing="ij"), -1).reshape(-1, 3)
    attrs = vol.reshape(-1, 6)
    origin, vs = np.array([-0.5, -0.5, -0.5], np.float32), 1 / (D - 1)
    pts = (origin + rng.random((500, 3)) * (D - 1) * vs).astype(np.float32)    # strictly inside
    got, covered = sparse_trilinear(coords, attrs, origin, vs, pts)
    assert covered.all()
    assert np.allclose(got, _dense_reference(vol, origin, vs, pts), atol=1e-5)


def test_sparse_grid_renormalises_at_missing_corners():
    # single voxel at (2,2,2) holding colour 1; a point between it and an empty neighbour
    coords = np.array([[2, 2, 2]]); attrs = np.ones((1, 3), np.float32)
    pts = np.array([[2.25, 2.0, 2.0]], np.float32)                      # 75% weight on the voxel
    got, cov = sparse_trilinear(coords, attrs, np.zeros(3), 1.0, pts)
    assert cov[0] and np.allclose(got[0], 1.0)                          # renormalised
    got, _ = sparse_trilinear(coords, attrs, np.zeros(3), 1.0, pts, renormalize=False)
    assert np.allclose(got[0], 0.75)                                    # official: missing -> zero
    got, cov = sparse_trilinear(coords, attrs, np.zeros(3), 1.0, np.array([[7.0, 7.0, 7.0]], np.float32))
    assert not cov[0] and np.allclose(got[0], 0)


def test_chunking_is_transparent():
    rng = np.random.default_rng(1)
    coords = np.unique(rng.integers(0, 12, (600, 3)), axis=0)
    attrs = rng.random((len(coords), 4)).astype(np.float32)
    pts = rng.random((1000, 3)).astype(np.float32) * 12
    a, ca = sparse_trilinear(coords, attrs, np.zeros(3), 1.0, pts, chunk=7)
    b, cb = sparse_trilinear(coords, attrs, np.zeros(3), 1.0, pts, chunk=10 ** 6)
    assert np.allclose(a, b) and np.array_equal(ca, cb)


def test_bake_texture_trilinear_path(tmp_path):
    """End to end through bake_texture: planar quad over a uniform-colour voxel slab."""
    from backends.texture_baker import bake_texture, build_textured_mesh
    verts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], np.float32) * 0.5 - 0.25
    faces = np.array([[0, 1, 2], [0, 2, 3]]); uvs = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32)
    g = np.arange(-4, 5)
    coords = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    attrs = np.tile(np.array([0.2, 0.6, 0.9, 0.0, 0.5, 1.0], np.float32), (len(coords), 1))
    vs = 1 / 8
    base, mr, mask = bake_texture(verts, faces, uvs, coords, attrs, np.zeros(3), vs, texture_size=32,
                                  sampling="trilinear", gamma=None)
    assert mask.sum() > 600
    px = base[mask].astype(float) / 255
    assert np.allclose(px.mean(axis=0), [0.2, 0.6, 0.9], atol=0.01)       # linear, uniform
    assert abs(mr[mask][:, 1].mean() / 255 - 0.5) < 0.01 and mr[mask][:, 2].max() == 0
    base2, _, _ = bake_texture(verts, faces, uvs, coords, attrs, np.zeros(3), vs, texture_size=32,
                               sampling="trilinear", gamma=2.2)
    assert base2[mask].mean() > base[mask].mean()                        # gamma brightened
    m = build_textured_mesh(verts, faces, uvs, base, mr)
    assert m.visual.material.baseColorTexture.size == (32, 32)
