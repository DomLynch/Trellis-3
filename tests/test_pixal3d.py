import json
import os

import numpy as np
import pytest
from PIL import Image

import generate
from trellis3 import pixal3d


def test_camera_matrices_match_pixal3d_example():
    """Matrices copied from Pixal3D assets/mv_images/example/transforms.json (azim 0/90/180/270)."""
    d = pixal3d.DISTANCE
    want = {
        0: [[1, 0, 0, 0], [0, 0, -1, -d], [0, 1, 0, 0], [0, 0, 0, 1]],
        90: [[0, 0, 1, d], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
        180: [[-1, 0, 0, 0], [0, 0, 1, d], [0, 1, 0, 0], [0, 0, 0, 1]],
        270: [[0, 0, -1, -d], [-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
    }
    for yaw, m in want.items():
        assert np.allclose(pixal3d.camera_matrix(yaw), m, atol=1e-9), yaw


def _rgba(path, w=300, h=500):
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    im.paste(Image.new("RGBA", (w // 2, h // 2), (200, 50, 50, 255)), (w // 4, h // 4))
    im.save(path)
    return str(path)


def test_write_views_dir_frames_square_and_poses(tmp_path):
    imgs = [_rgba(tmp_path / f"{i}.png") for i in range(3)]
    d = pixal3d.write_views_dir(imgs, [0, 45, 180], str(tmp_path / "v"))
    meta = json.load(open(os.path.join(d, "transforms.json")))
    assert [f["file_path"] for f in meta["frames"]] == ["view00_azim000.png", "view01_azim045.png", "view02_azim180.png"]
    im = Image.open(os.path.join(d, "view00_azim000.png"))
    assert im.size == (1024, 1024)
    h = im.getchannel("A").getbbox()
    assert (h[3] - h[1]) == pytest.approx(1024 * pixal3d.FILL, abs=2)


def test_first_view_must_be_front_and_alpha_required(tmp_path):
    a = _rgba(tmp_path / "a.png")
    with pytest.raises(ValueError, match="yaw 0"):
        pixal3d.write_views_dir([a, a], [90, 0], str(tmp_path / "v"))
    Image.new("RGB", (8, 8)).save(tmp_path / "rgb.png")
    with pytest.raises(ValueError, match="alpha"):
        pixal3d.write_views_dir([str(tmp_path / "rgb.png")], [0], str(tmp_path / "v2"))


def test_build_command_modes():
    single = pixal3d.build_command(["a.png"], "out/h")
    assert "--image" in single and "--views-dir" not in single and single[single.index("--output") + 1] == "out/h.glb"
    multi = pixal3d.build_command(["a.png", "b.png", "c.png"], "out/h", views_dir="vd")
    assert multi[multi.index("--num-views") + 1] == "3" and multi[multi.index("--views-dir") + 1] == "vd"


def test_default_yaws():
    assert pixal3d.default_yaws(4) == [0, 45, 90, 180]


def test_generate_dispatches_pixal3d_preset(tmp_path, monkeypatch):
    img = tmp_path / "i.png"
    Image.new("RGB", (4, 4)).save(img)
    seen = {}
    monkeypatch.setattr(pixal3d, "run_from_args", lambda a: seen.update(images=a.images, yaws=a.yaws) or 0)
    with pytest.raises(SystemExit) as e:
        generate.main(["--images", str(img), str(img), "--yaws", "0", "90", "--preset", "pixal3d", "--allow-battery",
                       "--output", str(tmp_path / "o")])
    assert e.value.code == 0 and seen["yaws"] == [0.0, 90.0] and len(seen["images"]) == 2


def test_yaws_count_must_match(tmp_path):
    img = tmp_path / "i.png"
    Image.new("RGB", (4, 4)).save(img)
    with pytest.raises(SystemExit):
        generate.main(["--images", str(img), str(img), "--yaws", "0", "--output", str(tmp_path / "o")])
