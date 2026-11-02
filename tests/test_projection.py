import numpy as np
import pytest

pytest.importorskip("cv2")
pytest.importorskip("scipy")

from trellis3 import projection as pj


def test_view_basis_conventions():
    r, up, d = pj.view_basis(0)
    assert np.allclose(r, [1, 0, 0]) and np.allclose(d, [0, 0, 1])
    r, up, d = pj.view_basis(90)                     # camera on +X, face (+Z) at image-left
    assert np.allclose(d, [1, 0, 0])
    x, y, z = pj.project(np.array([[0, 0, 1.0]]), np.zeros(3), 90, 100, 0, 0, 0, 0)
    assert x[0] < -99
    r, up, d = pj.view_basis(180)
    assert np.allclose(d, [0, 0, -1]) and np.allclose(r, [-1, 0, 0])


def test_rasterize_uv_quad_barycentrics():
    uv = np.array([[[0, 0], [1, 0], [0, 1]], [[1, 0], [1, 1], [0, 1]]], float)
    faces = np.array([[0, 1, 2], [1, 3, 2]])
    ids, ys, xs, f, bary = pj.rasterize_uv(uv, faces, 16, 16)
    assert len(ys) > 200                            # nearly the whole 16x16 square
    assert np.allclose(bary.sum(1), 1.0) and (bary >= 0).all()
    assert set(f) == {0, 1}


def test_blend_unseen_keeps_base_seen_takes_view():
    base = np.full((2, 3), 100.0)
    red = np.tile([200.0, 0, 0], (2, 1))
    out, conf = pj.blend(base, [red], [np.array([0.5, 0.0])], tau=0.25, sharp=1.0)
    assert np.allclose(out[1], base[1]) and conf[1] == 0       # unseen texel keeps the base colour
    assert np.allclose(out[0], red[0]) and conf[0] == 1        # confident texel is fully repainted


def test_blend_weak_view_is_partial_not_drowned():
    base = np.full((1, 3), 100.0)
    view = np.full((1, 3), 200.0)
    out, conf = pj.blend(base, [view], [np.array([0.05])], tau=0.1, sharp=1.0)
    assert conf[0] == pytest.approx(0.5) and out[0, 0] == pytest.approx(150.0)


def test_blend_two_views_weighted_average():
    base = np.zeros((1, 3))
    a, b = np.full((1, 3), 40.0), np.full((1, 3), 80.0)
    out, _ = pj.blend(base, [a, b], [np.ones(1), np.ones(1)], tau=0.1, sharp=1.0)
    assert np.allclose(out, 60.0)
    out, _ = pj.blend(base, [a, b], [np.full(1, 1.0), np.full(1, 3.0)], tau=0.1, sharp=1.0)
    assert out[0, 0] == pytest.approx((40 + 240) / 4)


def test_facing_weight_monotonic_and_zero_behind():
    d = np.array([0, 0, 1.0])
    n = np.array([[0, 0, 1.0], [0, 0.7071, 0.7071], [0, 0, -1.0]])
    w = pj.facing_weight(n, d, signed=True)
    assert w[0] == 1.0 and 0 < w[1] < 1 and w[2] == 0
    w2 = pj.facing_weight(n, d)             # default ignores the sign of unreliable normals
    assert w2[2] == 1.0 and w2[0] == 1.0


def test_depth_visible_hides_far_point():
    x = np.array([10.0, 10.0])
    y = np.array([10.0, 10.0])
    z = np.array([1.0, 0.0])                         # second point is behind the first on the same pixel
    vis = pj.depth_visible(x, y, z, (32, 32), extent=1.0)
    assert vis.tolist() == [True, False]


def test_fit_view_recovers_shift_and_scale():
    rng = np.random.default_rng(1)
    pts = rng.normal(size=(60000, 3)) * [0.25, 0.5, 0.15]
    pts = pts[np.abs(pts[:, 1]) < 0.5]
    centre = (pts.min(0) + pts.max(0)) / 2
    cw = 256
    sil = pj._silhouette(pts, centre, 0, 200.0, 20.0, -10.0, cw)
    import cv2
    alpha = (cv2.resize(sil.astype(np.uint8), (cw * 2, 512), interpolation=cv2.INTER_NEAREST) * 255).astype(np.uint8)
    fit = pj.fit_view(pts, alpha, yaw0=0.0)
    assert fit["iou"] > 0.9
    assert abs(fit["yaw"]) < 6
