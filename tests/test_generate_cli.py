import json
import os

import pytest
import torch
from PIL import Image

import generate
from trellis3.stages import MemoryPressure


def test_parser_accepts_new_options():
    a = generate.build_parser().parse_args(
        ["img.png", "--preset", "master", "--pipeline-type", "1536_cascade", "--texture-size", "8192",
         "--max-tokens", "20000", "--resident", "--tex-steps", "8", "--two-stage", "--min-free-gb", "3"])
    assert a.preset == "master" and a.pipeline_type == "1536_cascade" and a.texture_size == 8192
    assert a.tex_steps == 8 and a.two_stage and a.min_free_gb == 3.0 and a.gamma == "auto"


def test_parser_rejects_bad_type():
    with pytest.raises(SystemExit):
        generate.build_parser().parse_args(["img.png", "--pipeline-type", "2048"])


def test_gamma_arg_accepts_auto_and_float():
    p = generate.build_parser()
    assert p.parse_args(["i.png", "--gamma", "auto"]).gamma == "auto"
    assert p.parse_args(["i.png", "--gamma", "1.7"]).gamma == 1.7


class _Mesh:
    vertices = torch.zeros((3, 3))
    faces = torch.tensor([[0, 1, 2]], dtype=torch.int32)
    attrs = None


@pytest.fixture
def img(tmp_path):
    p = tmp_path / "in.png"
    Image.new("RGB", (4, 4)).save(p)
    return str(p)


@pytest.fixture
def fake(monkeypatch):
    seen = {"runners": []}

    class FakeRunner:
        def __init__(self, types, resident=False, stage_hooks=True, decode_only=False, device="auto"):
            seen["runners"].append({"types": types, "decode_only": decode_only})

        def sample(self, image, pipeline_type, seed, steps, max_num_tokens, state, on_stage, release,
                   min_free_gb, fusion="multidiffusion"):
            seen.update(fusion=fusion, n_images=len(image) if isinstance(image, list) else 1,
                        pipeline_type=pipeline_type, seed=seed, steps=steps, tokens=max_num_tokens,
                        release=release, min_free_gb=min_free_gb, resumed=state is not None)
            st = {"pipeline_type": pipeline_type, "seed": seed, "done": [], "res": 512,
                  "shape_feats": torch.zeros(2, 2), "shape_coords": torch.zeros(2, 4), "tex_feats": torch.zeros(2, 2)}
            for n in ("coords", "shape", "tex"):
                st["done"].append(n)
                if seen.get("fail") == n:
                    raise MemoryPressure("low")
                if on_stage:
                    on_stage(n, st)
            return st

        def decode(self, state):
            seen["decoded"] = state["done"]
            return _Mesh()

        def generate(self, *a, **k):
            seen["legacy"] = True
            return _Mesh()

    monkeypatch.setattr(generate, "Runner", FakeRunner)
    monkeypatch.setattr(generate, "wait_until_cool", lambda *a, **k: None)
    monkeypatch.setattr(generate, "export_mesh",
                        lambda mesh, out, **kw: seen.update(out=out, **kw) or {"faces": 1, "bake": "none"})
    return seen


def test_default_staged_run_saves_latents_and_receipt(img, tmp_path, fake):
    out = str(tmp_path / "o")
    generate.main([img, "--preset", "draft", "--seed", "7", "--output", out, "--max-tokens", "30000",
                   "--tex-steps", "4"])
    assert fake["pipeline_type"] == "512" and fake["seed"] == 7 and fake["tokens"] == 30000
    assert fake["bake_sampling"] == "trilinear"
    assert fake["steps"] == {"ss": 8, "shape": 8, "tex": 4} and fake["release"] is True
    assert fake["decoded"] == ["coords", "shape", "tex"]
    assert os.path.exists(out + ".latents.pt")
    assert fake["no_texture"] is True and fake["bake_faces"] == 200000 and fake["gamma"] == "auto"
    r = json.load(open(out + ".receipt.json"))
    assert r["settings"]["steps_per_stage"] == {"ss": 8, "shape": 8, "tex": 4}
    assert r["settings"]["latents"] == out + ".latents.pt"


def test_master_preset(img, tmp_path, fake, monkeypatch):
    calls = []
    monkeypatch.setattr(generate.subprocess, "run", lambda cmd: calls.append(cmd) or type("R", (), {"returncode": 0})())
    out = str(tmp_path / "m")
    with pytest.raises(SystemExit) as e:
        generate.main([img, "--preset", "master", "--output", out])
    assert e.value.code == 0
    assert fake["pipeline_type"] == "1024_cascade"
    cmd = calls[0]
    assert cmd[1].endswith("generate.py") and "--from-latents" in cmd and cmd[-1] == out + ".latents.pt"
    assert "--preset" in cmd and "master" in cmd        # child re-parses the same args
    assert "decoded" not in fake                        # parent did not decode


def test_from_latents_decode_only(img, tmp_path, fake):
    from trellis3.latents import save_state
    lp = str(tmp_path / "l.pt")
    save_state(lp, {"pipeline_type": "1024_cascade", "seed": 1, "done": ["coords", "shape", "tex"], "res": 1024,
                    "shape_feats": torch.zeros(1, 1), "shape_coords": torch.zeros(1, 4), "tex_feats": torch.zeros(1, 1)})
    generate.main([img, "--from-latents", lp, "--output", str(tmp_path / "d"), "--preset", "master"])
    assert fake["runners"][-1] == {"types": "1024_cascade", "decode_only": True}
    assert fake["decoded"] == ["coords", "shape", "tex"]
    assert fake["normal_map"] is True and fake["bake_faces"] == 1_000_000 and fake["texture_size"] == 4096
    r = json.load(open(str(tmp_path / "d.receipt.json")))
    assert r["settings"]["from_latents"] == lp


def test_from_incomplete_latents_exits_4(img, tmp_path, fake):
    from trellis3.latents import save_state
    lp = str(tmp_path / "l.pt")
    save_state(lp, {"pipeline_type": "512", "seed": 1, "done": ["coords"]})
    with pytest.raises(SystemExit) as e:
        generate.main([img, "--from-latents", lp])
    assert e.value.code == 4


def test_memory_pressure_exits_3_with_partial_latents(img, tmp_path, fake):
    fake["fail"] = "shape"
    out = str(tmp_path / "p")
    with pytest.raises(SystemExit) as e:
        generate.main([img, "--output", out, "--min-free-gb", "3"])
    assert e.value.code == 3 and fake["min_free_gb"] == 3.0
    from trellis3.latents import load_state
    assert load_state(out + ".latents.pt")["done"] == ["coords"]


def test_resume_passes_state(img, tmp_path, fake):
    from trellis3.latents import save_state
    out = str(tmp_path / "r")
    save_state(out + ".latents.pt", {"pipeline_type": "512", "seed": 42, "done": ["coords"]})
    generate.main([img, "--output", out, "--resume"])
    assert fake["resumed"] is True


def test_legacy_run(img, tmp_path, fake):
    generate.main([img, "--output", str(tmp_path / "x"), "--legacy-run"])
    assert fake["legacy"] is True


def test_main_missing_image_exits(tmp_path):
    with pytest.raises(SystemExit) as e:
        generate.main([str(tmp_path / "nope.png")])
    assert e.value.code == 1


def test_images_option_passes_all_views_and_fusion(img, tmp_path, fake):
    out = str(tmp_path / "mv")
    generate.main(["--images", img, img, img, "--fusion", "stochastic", "--output", out])
    assert fake["n_images"] == 3 and fake["fusion"] == "stochastic"
    r = json.load(open(out + ".receipt.json"))
    assert r["settings"]["images"] == [img, img, img] and r["settings"]["fusion"] == "stochastic"


def test_single_image_receipt_has_no_multiview_keys(img, tmp_path, fake):
    out = str(tmp_path / "one")
    generate.main([img, "--output", out])
    assert fake["n_images"] == 1
    r = json.load(open(out + ".receipt.json"))
    assert "images" not in r["settings"] and "fusion" not in r["settings"]


def test_positional_plus_images_makes_positional_first(img, tmp_path, fake):
    generate.main([img, "--images", img, "--output", str(tmp_path / "p")])
    assert fake["n_images"] == 2
