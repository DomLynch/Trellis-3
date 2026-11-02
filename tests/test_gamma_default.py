"""Gamma default is 1.0 (no colour transform); "auto" is refused under --preset hero. CPU only, no pipeline is built."""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
generate = pytest.importorskip("generate")


def test_default_is_no_transform():
    args = generate.build_parser().parse_args(["x.png"])
    assert args.gamma == 1.0
    assert generate.bake_kwargs(args, {"normal_map": False, "bake_faces": 1})["gamma"] == 1.0


def test_zero_is_an_explicit_one_not_none():
    args = generate.build_parser().parse_args(["x.png", "--gamma", "0"])
    assert generate.bake_kwargs(args, {"normal_map": False, "bake_faces": 1})["gamma"] == 1.0


def test_explicit_number_and_auto_still_parse_outside_hero():
    p = generate.build_parser()
    assert p.parse_args(["x.png", "--gamma", "1.7"]).gamma == 1.7
    assert p.parse_args(["x.png", "--gamma", "auto"]).gamma == "auto"


def test_auto_is_refused_under_hero(capsys):
    with pytest.raises(SystemExit) as e:
        generate.main(["x.png", "--preset", "hero", "--gamma", "auto"])
    assert e.value.code == 2
    assert "nothing automatic" in capsys.readouterr().err
