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


def test_accumulator_matches_blend():
    rng = np.random.default_rng(0)
    base = rng.uniform(0, 255, (50, 3))
    cols = [rng.uniform(0, 255, (50, 3)) for _ in range(3)]
    ws = [rng.uniform(0, 0.6, 50) for _ in range(3)]
    want, wconf = pj.blend(base, cols, ws, tau=0.2, sharp=2.0)
    acc = pj.Accumulator(50, 2.0)
    for c, w in zip(cols, ws):
        acc.add(c, w)
    got, gconf = acc.result(base, 0.2)
    assert np.allclose(got, want, atol=0.05) and np.allclose(gconf, wconf, atol=1e-5)


def _quad_glb(path):
    import trimesh
    from PIL import Image
    v = np.array([[-.5, -.5, 0], [.5, -.5, 0], [.5, .5, 0], [-.5, .5, 0]], float)
    f = np.array([[0, 1, 2], [0, 2, 3]])
    uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
    m = trimesh.Trimesh(v, f, process=False)
    m.visual = trimesh.visual.TextureVisuals(uv=uv, material=trimesh.visual.material.PBRMaterial(
        baseColorTexture=Image.new("RGB", (32, 32), (10, 10, 10))))
    m.export(path)


def _view(path, color):
    from PIL import Image
    im = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
    im.paste(Image.new("RGBA", (240, 240), color), (8, 8))
    im.save(path)


def test_project_texture_chunked_equals_single_chunk(tmp_path):
    from PIL import Image
    _quad_glb(str(tmp_path / 'q.glb'))
    _view(tmp_path / 'a.png', (200, 30, 0, 255))
    _view(tmp_path / 'b.png', (0, 60, 180, 255))
    views = [f"{tmp_path / 'a.png'}:0", f"{tmp_path / 'b.png'}:0:0.5"]
    outs = []
    for chunk in (4_000_000, 300):                     # one chunk vs many small chunks
        pj.project_texture(str(tmp_path / 'q.glb'), str(tmp_path / f'o{chunk}.glb'), views, fit=False, size=64,
                           out_png=str(tmp_path / f'o{chunk}.png'), chunk=chunk, log=lambda *_: None)
        outs.append(np.asarray(Image.open(tmp_path / f'o{chunk}.png')).astype(int))
    assert np.abs(outs[0] - outs[1]).max() <= 1


def test_project_texture_end_to_end_size_png_and_front_only(tmp_path):
    from PIL import Image
    _quad_glb(str(tmp_path / "q.glb"))
    _view(tmp_path / "front.png", (200, 0, 0, 255))
    _view(tmp_path / "other.png", (0, 0, 200, 255))
    mask = Image.new("L", (64, 64), 0)
    mask.paste(255, (0, 0, 32, 64))                     # left half of the texture is front-only
    mask.save(tmp_path / "mask.png")
    stat = pj.project_texture(str(tmp_path / "q.glb"), str(tmp_path / "o.glb"),
                              [f"{tmp_path / 'front.png'}:0", f"{tmp_path / 'other.png'}:0"], fit=False, size=64,
                              out_png=str(tmp_path / "o.png"), front_only_mask=str(tmp_path / "mask.png"),
                              log=lambda *_: None)
    out = np.asarray(Image.open(tmp_path / "o.png").convert("RGB")).astype(int)
    assert out.shape == (64, 64, 3) and stat["texture"] == [64, 64]
    left, right = out[32, 16], out[32, 48]
    assert left[0] > 150 and left[2] < 40               # front-only texel: red only
    assert right[0] > 20 and right[2] > 20              # shared texel: red and blue blended


def test_parse_view_spec_windows_paths_and_weights():
    assert pj.parse_view_spec(r"C:\a\b.png:90") == (r"C:\a\b.png", 90.0, 1.0)
    assert pj.parse_view_spec(r"C:\a\b.png:90:0.6") == (r"C:\a\b.png", 90.0, 0.6)
    assert pj.parse_view_spec("C:/a/b.png") == ("C:/a/b.png", 0.0, 1.0)
    assert pj.parse_view_spec("x.png:270:0.5") == ("x.png", 270.0, 0.5)
