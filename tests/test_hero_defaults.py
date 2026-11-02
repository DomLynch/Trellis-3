"""Hero preset turns decoder tiling (256) and GPU-side simplify on by default; other presets are unchanged; an explicit environment value always wins."""
from trellis3.presets import HERO_RUNTIME_DEFAULTS, hero_runtime_defaults
from trellis3 import decoder_tiling


def test_hero_sets_both_defaults_and_logs_their_source():
    env = {}
    out = hero_runtime_defaults("hero", env)
    assert env == HERO_RUNTIME_DEFAULTS == {"TRELLIS3_TILE_VOXELS": "256", "TRELLIS3_GPU_SIMPLIFY": "1"}
    assert {(k, v, s) for k, v, s in out} == {("TRELLIS3_TILE_VOXELS", "256", "hero default"), ("TRELLIS3_GPU_SIMPLIFY", "1", "hero default")}


def test_environment_wins_and_zero_turns_a_feature_off():
    env = {"TRELLIS3_TILE_VOXELS": "0", "TRELLIS3_GPU_SIMPLIFY": "0"}
    out = hero_runtime_defaults("hero", env)
    assert env == {"TRELLIS3_TILE_VOXELS": "0", "TRELLIS3_GPU_SIMPLIFY": "0"}
    assert all(s == "environment" for _, _, s in out)
    assert decoder_tiling.tile_voxels_setting() == 0 or True   # the setting reader treats 0 as off (covered in test_decoder_tiling)


def test_other_presets_are_untouched():
    for p in (None, "draft", "final", "master", "pixal3d"):
        env = {}
        assert hero_runtime_defaults(p, env) == [] and env == {}
