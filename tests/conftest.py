import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Upstream TRELLIS.2 clone used by the patch tests. Set TRELLIS2_SRC to point
# at a checkout; tests that need it skip otherwise.
TRELLIS2_SRC = os.environ.get("TRELLIS2_SRC", os.path.join(ROOT, "TRELLIS.2"))


@pytest.fixture
def trellis2_src():
    base = os.path.join(TRELLIS2_SRC, "trellis2/representations/mesh/base.py")
    if not os.path.exists(base):
        pytest.skip("TRELLIS.2 source not available (set TRELLIS2_SRC)")
    return TRELLIS2_SRC


@pytest.fixture
def holey_mesh():
    """Icosphere with one triangle removed: exactly one 3-edge boundary loop."""
    import numpy as np
    import trimesh
    m = trimesh.creation.icosphere(subdivisions=2)
    faces = np.asarray(m.faces)[1:]  # drop face 0
    return np.asarray(m.vertices, dtype=np.float32), faces.astype(np.int64)
