import numpy as np
import trimesh

from trellis3 import hires


def _sphere(sub, r=0.5):
    m = trimesh.creation.icosphere(subdivisions=sub, radius=r)
    return np.asarray(m.vertices, np.float32), np.asarray(m.faces, np.int64), m


def test_closest_point_matches_trimesh():
    v, f, m = _sphere(3)
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(2000, 3)).astype(np.float32) * 0.4
    ns = hires.NearestSurface(v, f, k=8, exact=True)
    cp, fid, bary, d = ns.query(pts)
    ref_cp, ref_d, ref_f = trimesh.proximity.closest_point(m, pts)
    assert np.allclose(d, ref_d, atol=2e-5)
    assert np.allclose(cp, ref_cp, atol=2e-4)
    # barycentric reconstructs the closest point
    rec = (v[f[fid]] * bary[..., None]).sum(axis=1)
    assert np.allclose(rec, cp, atol=1e-5)
    assert np.all(bary >= -1e-5) and np.allclose(bary.sum(1), 1, atol=1e-5)


def test_points_on_surface_have_zero_distance():
    v, f, _ = _sphere(2)
    ns = hires.NearestSurface(v, f, exact=True)
    centroids = v[f].mean(axis=1)
    _, fid, _, d = ns.query(centroids)
    assert np.allclose(d, 0, atol=1e-6)
    assert np.array_equal(fid, np.arange(len(f)))


def test_normals_point_outward_and_smooth():
    v, f, _ = _sphere(3)
    ns = hires.NearestSurface(v, f)
    pts = v[f].mean(axis=1)
    _, fid, bary, _ = ns.query(pts)
    n_flat = ns.normal(fid, bary, smooth=False)
    n_smooth = ns.normal(fid, bary, smooth=True)
    radial = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    assert (n_flat * radial).sum(1).min() > 0.95
    assert (n_smooth * radial).sum(1).min() > 0.99  # smooth is closer to the true sphere normal


def test_query_handles_k_larger_than_faces():
    tri_v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32)
    ns = hires.NearestSurface(tri_v, np.array([[0, 1, 2]]), k=8, exact=True)
    cp, fid, bary, d = ns.query(np.array([[0.2, 0.2, 1.0]], np.float32))
    assert np.allclose(cp, [0.2, 0.2, 0]) and fid[0] == 0 and abs(d[0] - 1) < 1e-6


def _unwrapped_low_sphere():
    """Low-res sphere with a trivial planar UV (good enough for a tangent frame)."""
    v, f, _ = _sphere(2)
    uv = np.stack([np.arctan2(v[:, 1], v[:, 0]) / (2 * np.pi) + 0.5,
                   np.arccos(np.clip(v[:, 2] / 0.5, -1, 1)) / np.pi], axis=1).astype(np.float32)
    # drop faces that wrap around the seam so the UV triangles are sane
    keep = np.ptp(uv[f][:, :, 0], axis=1) < 0.5
    return v, f[keep], uv


def test_normal_map_of_sphere_is_mostly_flat_blue():
    lv, lf, luv = _unwrapped_low_sphere()
    hv, hf, _ = _sphere(5)
    ns = hires.NearestSurface(hv, hf)
    img, mask = hires.bake_normal_map(lv, lf, luv, ns, texture_size=128, log=lambda *_: None)
    assert img.shape == (128, 128, 3) and mask.sum() > 1000
    px = img[mask].astype(float)
    assert px[:, 2].mean() > 235                   # mostly +Z
    assert abs(px[:, 0].mean() - 127.5) < 12 and abs(px[:, 1].mean() - 127.5) < 12
    assert img[~mask][:, 2].min() >= 200           # background stays "up" after dilation


def test_normal_map_captures_hires_detail():
    lv, lf, luv = _unwrapped_low_sphere()
    hv, hf, _ = _sphere(5)
    bump = 1 + 0.03 * np.sin(40 * hv[:, 0]) * np.sin(40 * hv[:, 1])
    hv = (hv * bump[:, None]).astype(np.float32)
    ns = hires.NearestSurface(hv, hf)
    img, mask = hires.bake_normal_map(lv, lf, luv, ns, texture_size=128, log=lambda *_: None)
    px = img[mask].astype(float)
    assert px[:, 0].std() > 15 and px[:, 1].std() > 15  # visible variation, not flat


def test_glb_frame_roundtrip():
    v = np.random.default_rng(1).normal(size=(10, 3)).astype(np.float32)
    g = v.copy()
    g[:, 1], g[:, 2] = v[:, 2].copy(), -v[:, 1].copy()   # to_glb's conversion
    assert np.allclose(hires.glb_to_decoder_frame(g), v)


def test_attach_normal_map():
    m = trimesh.creation.box()
    m.visual = trimesh.visual.TextureVisuals(
        uv=np.zeros((len(m.vertices), 2)), material=trimesh.visual.material.PBRMaterial())
    img = np.full((4, 4, 3), 128, np.uint8)
    assert hires.attach_normal_map(m, img) == 1
    assert m.visual.material.normalTexture.size == (4, 4)


def test_vertex_mode_is_close_to_exact_on_dense_mesh():
    v, f, m = _sphere(5)                       # edge ~0.027
    edge = np.linalg.norm(v[f[:, 0]] - v[f[:, 1]], axis=1).max()
    rng = np.random.default_rng(2)
    pts = rng.normal(size=(5000, 3)).astype(np.float32)
    pts *= 0.5 / np.linalg.norm(pts, axis=1, keepdims=True)   # on the true sphere
    fast = hires.NearestSurface(v, f)
    cp, fid, bary, d = fast.query(pts)
    assert d.max() <= edge * 0.6                 # within ~half an edge
    assert np.all(bary.sum(1) == 1) and np.all(bary.max(1) == 1)
    rec = (v[f[fid]] * bary[..., None]).sum(axis=1)
    assert np.allclose(rec, cp)
    n_fast = fast.normal(fid, bary)
    ref_cp, _, ref_f = trimesh.proximity.closest_point(m, pts)
    radial = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    assert (n_fast * radial).sum(1).min() > 0.98
    assert np.linalg.norm(cp - ref_cp, axis=1).max() <= edge * 0.6
