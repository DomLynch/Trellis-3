import pytest
import torch
from PIL import Image

import generate


def test_parser_accepts_new_options():
    a = generate.build_parser().parse_args(
        ["img.png", "--preset", "draft", "--pipeline-type", "1536_cascade",
         "--texture-size", "4096", "--max-tokens", "20000", "--resident"])
    assert a.preset == "draft" and a.pipeline_type == "1536_cascade"
    assert a.texture_size == 4096 and a.max_tokens == 20000 and a.resident


def test_parser_rejects_bad_type():
    with pytest.raises(SystemExit):
        generate.build_parser().parse_args(["img.png", "--pipeline-type", "2048"])


class _Mesh:
    vertices = torch.zeros((3, 3))
    faces = torch.tensor([[0, 1, 2]], dtype=torch.int32)
    attrs = None


def test_main_wiring(tmp_path, monkeypatch):
    img = tmp_path / "in.png"
    Image.new("RGB", (4, 4)).save(img)
    seen = {}

    class FakeRunner:
        def __init__(self, types, resident=False, stage_hooks=True):
            seen["types"] = types
            seen["resident"] = resident

        def generate(self, image, seed, pipeline_type, steps, max_num_tokens):
            seen.update(seed=seed, pipeline_type=pipeline_type, steps=steps, tokens=max_num_tokens)
            return _Mesh()

    monkeypatch.setattr(generate, "Runner", FakeRunner)
    monkeypatch.setattr(generate, "wait_until_cool", lambda *a, **k: seen.setdefault("thermal", True))
    monkeypatch.setattr(generate, "export_mesh",
                        lambda mesh, out, **kw: seen.update(out=out, **kw))

    generate.main([str(img), "--preset", "draft", "--seed", "7", "--output", str(tmp_path / "o"),
                   "--max-tokens", "30000"])
    assert seen["types"] == "512" and seen["pipeline_type"] == "512"
    assert seen["steps"] == 8 and seen["no_texture"] is True
    assert seen["seed"] == 7 and seen["tokens"] == 30000 and seen["thermal"] is True


def test_main_missing_image_exits(tmp_path):
    with pytest.raises(SystemExit) as e:
        generate.main([str(tmp_path / "nope.png")])
    assert e.value.code == 1
