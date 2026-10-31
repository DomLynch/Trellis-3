import pytest

from trellis3.presets import apply_preset


def test_no_preset_is_upstream_default():
    assert apply_preset(None) == {"pipeline_type": "512", "steps": None,
                                  "texture_size": 1024, "no_texture": False}


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
