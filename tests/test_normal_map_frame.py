"""Normal-map frame: bake a subdivided sphere onto a coarse decimated copy of itself and rebuild the shading normal the way a glTF renderer does (interpolated vertex normal +
UV-derived tangent). A map encoded against the flat face normal prints the low-poly layout (large angular error); the smooth-frame bake reproduces the sphere. CPU only."""
import numpy as np
import pytest
import trimesh

from trellis3 import hires as hr

xatlas = pytest.importorskip("xatlas")


def _setup(sub_low=1, sub_hi=5, size=192):
    from backends.texture_baker import uv_unwrap
    low = trimesh.creation.icosphere(subdivisions=sub_low, radius=1.0)
    hi = trimesh.creation.icosphere(subdivisions=sub_hi, radius=1.0)
    nv, nf, uvs, _ = uv_unwrap(np.asarray(low.vertices, np.float32), np.asarray(low.faces))
    index = hr.NearestSurface(np.asarray(hi.vertices, np.float32), np.asarray(hi.faces), k=8)
    return nv, nf, uvs, index, size


def _render_error(nv, nf, uvs, img, mask, size, v_down):
    """Angle (deg) between the normal a renderer reconstructs from the map and the TRUE sphere normal, over the covered texels."""
    positions, m2, face_ids = hr._rasterize(nv, nf, uvs, size, v_down)
    assert (m2 == mask).all()
    lf, p = face_ids[mask], positions[mask]
    # the renderer's inputs, built independently of the baker: exact smooth sphere vertex normals, per-vertex tangents from the UV derivatives
    Nv = nv / np.linalg.norm(nv, axis=1, keepdims=True)
    Tv, Hv = hr._vertex_tangents(nv, nf, uvs, Nv.astype(np.float32))
    bary = hr._barycentric(p, nv[nf[lf]])
    fl = nf[lf]
    n = (Nv[fl] * bary[..., None]).sum(1); n /= np.linalg.norm(n, axis=1, keepdims=True)
    t = (Tv[fl] * bary[..., None]).sum(1); t -= n * (t * n).sum(1, keepdims=True); t /= np.linalg.norm(t, axis=1, keepdims=True)
    h = np.where((Hv[fl] * bary[..., None]).sum(1) >= 0, 1.0, -1.0)
    b = np.cross(n, t) * h
    ts = img[mask].astype(np.float32) / 255 * 2 - 1
    shaded = ts[:, :1] * t + ts[:, 1:2] * b + ts[:, 2:3] * n
    shaded /= np.linalg.norm(shaded, axis=1, keepdims=True)
    truth = p / np.linalg.norm(p, axis=1, keepdims=True)
    return np.degrees(np.arccos(np.clip((shaded * truth).sum(1), -1, 1)))


def test_flat_frame_bake_prints_the_triangle_layout_and_the_smooth_frame_does_not():
    nv, nf, uvs, index, size = _setup()
    quiet = lambda *_: None
    smooth, mask = hr.bake_normal_map(nv, nf, uvs, index, size, v_down=True, log=quiet, shading="smooth")
    flat, mask2 = hr.bake_normal_map(nv, nf, uvs, index, size, v_down=True, log=quiet, shading="flat")
    e_smooth = _render_error(nv, nf, uvs, smooth, mask, size, True)
    e_flat = _render_error(nv, nf, uvs, flat, mask2, size, True)
    print(f"render error vs the true sphere normal: smooth p50/p95/max {np.percentile(e_smooth, [50, 95, 100]).round(2)}  flat {np.percentile(e_flat, [50, 95, 100]).round(2)}")
    assert np.percentile(e_flat, 95) > 8.0          # the facet angle of an 80-face sphere: the shards
    assert np.percentile(e_smooth, 95) < 3.0        # smooth frame: the map just adds the missing detail
    assert e_smooth.mean() < e_flat.mean() / 4


def test_default_is_the_smooth_frame_and_uses_given_vertex_normals():
    nv, nf, uvs, index, size = _setup(size=96)
    quiet = lambda *_: None
    default, mask = hr.bake_normal_map(nv, nf, uvs, index, size, v_down=True, log=quiet)
    explicit, _ = hr.bake_normal_map(nv, nf, uvs, index, size, v_down=True, log=quiet, low_normals=(nv / np.linalg.norm(nv, axis=1, keepdims=True)).astype(np.float32))
    assert np.abs(default.astype(int) - explicit.astype(int)).max() <= 12   # welded normals of the coarse sphere vs the exact sphere normals: close
    assert _render_error(nv, nf, uvs, default, mask, size, True).mean() < 3.0


def test_welded_vertex_normals_are_shared_across_uv_seams():
    nv, nf, uvs, index, size = _setup(size=64)
    n = hr.vertex_normals_welded(nv, nf)
    key = {}
    for i, v in enumerate(np.round(nv, 5)):
        key.setdefault(tuple(v), []).append(i)
    dup = [ids for ids in key.values() if len(ids) > 1]
    assert dup, "the unwrap should have split some vertices"
    for ids in dup:
        assert np.abs(n[ids] - n[ids[0]]).max() < 1e-5
