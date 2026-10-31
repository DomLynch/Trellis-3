import pytest
import torch

from trellis3 import latents


def test_roundtrip_and_describe(tmp_path):
    st = {"pipeline_type": "1024_cascade", "seed": 3, "done": ["coords", "shape"],
          "coords": torch.randint(0, 9, (50, 4)), "shape_feats": torch.randn(70, 32),
          "shape_coords": torch.randint(0, 9, (70, 4)), "res": 1024, "image_png": b"\x89PNG",
          "rng": {"cpu": torch.get_rng_state()}}
    p = latents.save_state(str(tmp_path / "a" / "x.latents.pt"), st)
    back = latents.load_state(p)
    assert back["done"] == ["coords", "shape"] and back["res"] == 1024
    assert torch.equal(back["shape_feats"], st["shape_feats"]) and back["image_png"] == b"\x89PNG"
    assert torch.equal(back["rng"]["cpu"], st["rng"]["cpu"])
    d = latents.describe(back)
    assert "voxels=50" in d and "tokens=70@1024" in d and "done=['coords', 'shape']" in d
    assert not (tmp_path / "a" / "x.latents.pt.tmp").exists()


def test_rejects_foreign_file(tmp_path):
    torch.save({"hello": 1}, tmp_path / "bad.pt")
    with pytest.raises(ValueError):
        latents.load_state(str(tmp_path / "bad.pt"))
