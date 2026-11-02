import numpy as np
import pytest

pytest.importorskip("scipy")

from trellis3 import registration as reg

CENTRE = np.zeros(3)
# a toy "face": 3D landmarks with real depth variation
PTS = np.array([[-0.07, 0.18, 0.22], [0.07, 0.18, 0.22], [0.0, 0.07, 0.31], [0.0, -0.01, 0.26],
                [0.0, -0.12, 0.23], [0.0, 0.22, 0.28]], float)


def test_solve_similarity_recovers_known_camera():
    x, y, _ = reg.project_roll(PTS, CENTRE, 0.0, 3000.0, 4.0, 2000.0, 1500.0)
    sol = reg.solve_camera(PTS, np.c_[x, y], CENTRE, 0.0, fix_yaw=True)
    assert sol["rms_px"] < 0.5 and sol["scale"] == pytest.approx(3000.0, rel=1e-2) and sol["roll"] == pytest.approx(4.0, abs=0.2)


def test_affine_plus_warp_lands_every_landmark_exactly():
    x, y, _ = reg.project_roll(PTS, CENTRE, 0.0, 3000.0, 0.0, 2000.0, 1500.0)
    img = np.c_[x * 1.15, y] + np.array([[12, -8], [-15, 20], [6, 6], [0, -25], [18, 0], [-9, 14]])   # wider face + noise
    sol = reg.solve_camera(PTS, img, CENTRE, 0.0, fix_yaw=True, affine=True)
    assert sol["kind"] == "affine" and sol["rms_px"] > 5                     # a pure affine cannot match it
    px, py, _ = reg.project_cam(PTS, sol, CENTRE)
    assert np.abs(px - img[:, 0]).max() < 1e-3 and np.abs(py - img[:, 1]).max() < 1e-3   # the TPS warp makes it exact


def test_mirror_picks_flips_x_and_swaps_labels():
    m = reg.mirror_picks({"eye_vl": [100, 50], "eye_vr": [300, 55], "nose_base": [200, 120]}, 400)
    assert m["eye_vr"] == [299, 50] and m["eye_vl"] == [99, 55] and m["nose_base"] == [199, 120]


def test_render_splat_and_lift_round_trip():
    rng = np.random.default_rng(0)
    pos = rng.uniform(-0.2, 0.2, (20000, 3)).astype(np.float32)
    pos[:, 2] = 0.1 + 0.05 * pos[:, 0]                          # a slanted sheet facing +Z
    col = np.full((20000, 3), 200, np.uint8)
    nrm = np.tile(np.array([0, 0, 1.0], np.float32), (20000, 1))
    img, dep, cam = reg.render_splat(pos, col, nrm, CENTRE, 0.0, size=400)
    target = pos[123]
    px = cam["scale"] * target[0] + cam["tx"]
    py = cam["scale"] * (-target[1]) + cam["ty"]
    p3 = reg.lift(px, py, dep, cam, CENTRE, 0.0)
    assert np.allclose(p3, target, atol=0.01)


def test_face_region_keeps_face_drops_hair_above_and_back_of_head():
    mesh3d = {"eye_vl": [-0.067, 0.18, 0.22], "eye_vr": [0.067, 0.18, 0.22]}
    pts = np.array([[0.0, 0.0, 0.25],      # cheek / mouth level, front
                    [0.0, 0.18, 0.22],     # between the eyes
                    [0.0, 0.60, 0.10],     # far above the forehead (hair)
                    [0.0, 0.10, -0.30],    # back of the head, same front-plane position as the face
                    [0.4, 0.1, 0.1]])      # far to the side
    w = reg.face_region_weight(pts, mesh3d, CENTRE)
    assert w[0] > 0.9 and w[1] > 0.9 and w[2] == 0 and w[3] == 0 and w[4] == 0
