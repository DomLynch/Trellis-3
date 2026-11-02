import numpy as np
import pytest

pytest.importorskip("cv2")

from trellis3 import materials


def _tex():
    t = np.zeros((64, 64, 3), np.uint8)
    t[:, :16] = (212, 175, 55)      # gold
    t[:, 16:32] = (20, 18, 18)      # hair / dark
    t[:, 32:48] = (225, 180, 160)   # skin
    t[:, 48:] = (120, 40, 50)       # red cloth
    return t


def test_classes_separate():
    m = materials.classify(_tex())
    assert m["gold"][:, :16].all() and not m["gold"][:, 16:].any()
    assert m["hair"][:, 16:32].all()
    assert m["skin"][:, 32:48].all()
    assert not (m["gold"] | m["hair"] | m["skin"])[:, 48:].any()


def test_metallic_roughness_values():
    mr, share = materials.metallic_roughness(_tex(), blur=0)
    g, b = mr[..., 1] / 255.0, mr[..., 2] / 255.0
    assert g[10, 8] == pytest.approx(0.28, abs=0.01) and b[10, 8] == pytest.approx(0.85, abs=0.01)
    assert g[10, 24] == pytest.approx(0.38, abs=0.01)
    assert g[10, 40] == pytest.approx(0.50, abs=0.01)
    assert g[10, 56] == pytest.approx(0.75, abs=0.01) and b[10, 56] == 0
    assert mr.shape == (64, 64, 3) and set(share) == {"gold", "hair", "skin"}
