import numpy as np
import trimesh
from PIL import Image

from trellis3 import postbake


def _textured_box():
    m = trimesh.creation.box()
    tex = Image.fromarray(np.full((4, 4, 3), 64, np.uint8))
    m.visual = trimesh.visual.TextureVisuals(
        uv=np.zeros((len(m.vertices), 2)),
        material=trimesh.visual.material.PBRMaterial(baseColorTexture=tex, metallicFactor=1.0),
    )
    return m


def test_set_metallic_factor_on_mesh_and_scene():
    m = _textured_box()
    assert postbake.set_metallic_factor(m, 0.0) == 1
    assert m.visual.material.metallicFactor == 0.0
    s = trimesh.Scene([_textured_box(), _textured_box()])
    assert postbake.set_metallic_factor(s, 0.25) == 2
    assert postbake.set_metallic_factor(m, None) == 0


def test_apply_gamma_brightens_and_keeps_alpha():
    m = _textured_box()
    assert postbake.apply_gamma(m, 2.2) == 1
    out = np.asarray(m.visual.material.baseColorTexture)
    assert out.shape[-1] == 4
    # 64/255 -> (0.251)^(1/2.2) -> ~0.534 -> ~136
    assert 130 <= out[0, 0, 0] <= 140
    assert out[0, 0, 3] == 255
    assert postbake.apply_gamma(m, None) == 0 and postbake.apply_gamma(m, 1.0) == 0


def test_fix_winding_repairs_flipped_half():
    ico = trimesh.creation.icosphere(subdivisions=2)
    faces = np.asarray(ico.faces).copy()
    half = len(faces) // 2
    faces[:half] = faces[:half, ::-1]  # flip half
    bad = trimesh.Trimesh(ico.vertices, faces, process=False)
    assert not bad.is_winding_consistent
    v, f, flipped = postbake.fix_winding(ico.vertices, faces)
    good = trimesh.Trimesh(v, f, process=False)
    assert good.is_winding_consistent and good.volume > 0
    assert flipped == half


def test_fix_winding_noop_on_good_mesh():
    ico = trimesh.creation.icosphere(subdivisions=1)
    _, _, flipped = postbake.fix_winding(ico.vertices, ico.faces)
    assert flipped == 0


def test_auto_gamma_matches_reference_mean():
    # texture mean 64/255 = 0.251; reference 0.50 -> g = log(.251)/log(.5) ~ 1.99
    m = _textured_box()
    ref = Image.fromarray(np.full((4, 4, 3), 128, np.uint8))
    g, mt, mr = postbake.auto_gamma(m, ref)
    assert abs(g - 1.99) < 0.03
    postbake.apply_gamma(m, g)
    out = np.asarray(m.visual.material.baseColorTexture)[..., :3].mean() / 255
    assert abs(out - 0.5) < 0.02


def test_auto_gamma_uses_reference_alpha_and_clamps():
    m = _textured_box()
    ref = np.zeros((4, 4, 4), np.uint8)
    ref[..., :3] = 255                 # white background...
    ref[:2, :, 3] = 255; ref[:2, :, :3] = 128   # ...but the foreground (alpha) is mid grey
    g, _, mr = postbake.auto_gamma(m, Image.fromarray(ref, "RGBA"))
    assert np.allclose(mr, 128 / 255, atol=1e-3) and abs(g - 1.99) < 0.03
    bright = Image.fromarray(np.full((2, 2, 3), 250, np.uint8))
    assert postbake.auto_gamma(m, bright)[0] == 3.0   # clamped to hi
    assert postbake.auto_gamma(trimesh.creation.box(), bright)[0] is None
