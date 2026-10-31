import pytest

from trellis3.presets import apply_preset


def test_no_preset_is_upstream_default():
    d = apply_preset(None)
    assert {k: d[k] for k in ("pipeline_type", "steps", "texture_size", "no_texture")} == {
        "pipeline_type": "512", "steps": None, "texture_size": 1024, "no_texture": False}


def test_draft_is_cheap():
    d = apply_preset("draft")
    assert d["pipeline_type"] == "512" and d["steps"] == 8 and d["no_texture"] is True


def test_final_is_full_quality():
    f = apply_preset("final")
    assert f["pipeline_type"] == "1024_cascade" and f["texture_size"] == 2048
    assert f["no_texture"] is False and f["steps"] is None


def test_explicit_flags_override_preset():
    d = apply_preset("draft", pipeline_type="1024_cascade", steps=12, texture_size=2048,
                     no_texture=False)
    # no_texture=False is the argparse default so it cannot override draft's True;
    # everything else wins.
    assert d["pipeline_type"] == "1024_cascade" and d["steps"] == 12
    assert d["texture_size"] == 2048
    assert d["no_texture"] is True


def test_no_texture_overrides_final():
    assert apply_preset("final", no_texture=True)["no_texture"] is True


def test_unknown_preset():
    with pytest.raises(ValueError):
        apply_preset("ultra")


def test_master_preset_and_new_keys():
    m = apply_preset("master")
    assert m["pipeline_type"] == "1024_cascade" and m["texture_size"] == 4096
    assert m["bake_faces"] == 1_000_000 and m["normal_map"] is True and m["two_stage"] is True
    d = apply_preset("draft")
    assert d["bake_faces"] == 200_000 and d["normal_map"] is False and d["two_stage"] is False
    o = apply_preset("master", bake_faces=300_000, texture_size=2048)
    assert o["bake_faces"] == 300_000 and o["texture_size"] == 2048 and o["normal_map"] is True


def test_hero_preset_and_single_process():
    h = apply_preset("hero")
    assert h["pipeline_type"] == "1536_cascade" and h["texture_size"] == 8192 and h["bake_faces"] == 2_000_000
    assert h["two_stage"] is True and h["normal_map"] is True
    assert apply_preset("hero", single_process=True)["two_stage"] is False
    assert apply_preset("master", single_process=True)["two_stage"] is False


def test_token_budget_defaults_by_machine_size(monkeypatch):
    from trellis3 import presets
    assert presets.default_max_tokens(total_gb=24) == 49152
    assert presets.default_max_tokens(total_gb=64) == 98304
    monkeypatch.setattr(presets, "default_max_tokens", lambda total_gb=None: 98304)
    assert apply_preset("hero")["max_num_tokens"] == 98304
    assert apply_preset("draft")["max_num_tokens"] == 98304          # machine-wide, not per preset
    assert apply_preset("hero", max_num_tokens=30000)["max_num_tokens"] == 30000
