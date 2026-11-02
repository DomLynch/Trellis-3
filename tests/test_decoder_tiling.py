"""Tiled shape decoder (trellis3.decoder_tiling): one tile = identical, many tiles = same rows in the same order within float noise.
Runs on CPU with the pure-torch sparse-convolution backend; skips when the upstream TRELLIS.2 source or torch is missing."""
import os
import sys

import pytest

torch = pytest.importorskip("torch")


@pytest.fixture(scope="module")
def sp_env():
    from conftest import TRELLIS2_SRC
    if not os.path.exists(os.path.join(TRELLIS2_SRC, "trellis2", "modules", "sparse", "basic.py")):
        pytest.skip("TRELLIS.2 source not available (set TRELLIS2_SRC)")
    os.environ.setdefault("SPARSE_CONV_BACKEND", "none")
    os.environ.setdefault("SPARSE_ATTN_BACKEND", "sdpa")
    os.environ.setdefault("ATTN_BACKEND", "sdpa")
    if TRELLIS2_SRC not in sys.path:
        sys.path.insert(0, TRELLIS2_SRC)
    try:
        from trellis2.modules import sparse as sp
        from trellis2.models.sc_vaes import sparse_unet_vae as m
    except Exception as e:   # no CPU-capable sparse backend in this environment
        pytest.skip(f"trellis2 sparse modules not importable here: {e!r}")
    if os.environ.get("SPARSE_CONV_BACKEND") != "none":
        pytest.skip("needs SPARSE_CONV_BACKEND=none")
    return sp, m


def make_decoder(m, num_blocks=(1, 1, 1, 1, 0)):
    torch.manual_seed(0)
    d = m.SparseUnetVaeDecoder(
        out_channels=7, model_channels=[8, 8, 8, 8, 8], latent_channels=4, num_blocks=list(num_blocks),
        block_type=["SparseConvNeXtBlock3d"] * 5, up_block_type=["SparseResBlockC2S3d"] * 4,
        block_args=[{}] * 5, use_fp16=False, pred_subdiv=True)
    for p in d.parameters():   # zero-initialised layers would make the test trivial
        p.data.normal_(0, 0.3)
    return d.eval()


def make_latent(sp, n_batch=1, side=8, density=0.35, seed=1):
    g = torch.Generator().manual_seed(seed)
    coords = []
    for b in range(n_batch):
        cells = torch.nonzero(torch.rand((side, side, side), generator=g) < density)
        coords.append(torch.cat([torch.full((cells.shape[0], 1), b), cells], dim=1))
    coords = torch.cat(coords).int()
    return sp.SparseTensor(torch.randn((coords.shape[0], 4), generator=g), coords)


def run(decoder, x, tile):
    from trellis3 import decoder_tiling
    decoder_tiling.install(decoder, tile)
    with torch.no_grad():
        return decoder(x, return_subs=True)


def test_off_and_one_tile_are_identical_and_many_tiles_match(sp_env):
    sp, m = sp_env
    dec = make_decoder(m)
    x = make_latent(sp)
    base, base_subs = run(dec, x, 0)
    assert base.coords.shape[0] > 500

    one, one_subs = run(dec, x, 4096)   # a single tile that covers the whole volume
    assert dec.last_tiles == 1
    assert torch.equal(one.coords, base.coords) and torch.equal(one.feats, base.feats)
    assert all(torch.equal(a.feats, b.feats) and torch.equal(a.coords, b.coords) for a, b in zip(one_subs, base_subs))
    assert len(one_subs) == len(base_subs) == 4

    many, many_subs = run(dec, x, 16)   # 4 x 4 x 4 cells at the stage-3 resolution
    assert dec.last_tiles > 8
    assert torch.equal(many.coords, base.coords)   # same rows in the same order
    assert (many.feats - base.feats).abs().max().item() <= 1e-4
    assert all(torch.equal(a.feats, b.feats) and torch.equal(a.coords, b.coords) for a, b in zip(many_subs, base_subs))


def test_two_batch_items_and_a_tile_that_cuts_through(sp_env):
    sp, m = sp_env
    dec = make_decoder(m, (1, 2, 1, 2, 0))
    x = make_latent(sp, n_batch=2, side=8, density=0.25, seed=3)
    base, _ = run(dec, x, 0)
    tiled, _ = run(dec, x, 8)
    assert dec.last_tiles > 16   # two batch items, 8-voxel cubes
    assert torch.equal(tiled.coords, base.coords)
    assert (tiled.feats - base.feats).abs().max().item() <= 1e-4


def test_guided_texture_decoder_one_tile_identical_and_many_tiles_match(sp_env):
    """The texture decoder is the same network with pred_subdiv=False, guided by the shape decoder's subdivisions."""
    sp, m = sp_env
    from trellis3 import decoder_tiling
    torch.manual_seed(5)
    shape_dec = make_decoder(m)
    tex = m.SparseUnetVaeDecoder(out_channels=6, model_channels=[8, 8, 8, 8, 8], latent_channels=4, num_blocks=[1, 1, 1, 1, 0],
                                 block_type=["SparseConvNeXtBlock3d"] * 5, up_block_type=["SparseResBlockC2S3d"] * 5, block_args=[{}] * 5,
                                 use_fp16=False, pred_subdiv=False).eval()
    for p in tex.parameters():
        p.data.normal_(0, 0.3)
    x = make_latent(sp)
    base, subs = run(shape_dec, x, 0)
    xt = x.replace(torch.randn((x.feats.shape[0], 4), generator=torch.Generator().manual_seed(9)))

    def run_tex(tile):
        decoder_tiling.install(tex, tile)
        with torch.no_grad():
            return tex(xt, guide_subs=subs)

    ref = run_tex(0)
    assert ref.coords.shape[0] == base.coords.shape[0]   # guided: same rows as the shape decode
    one = run_tex(4096)
    assert tex.last_tiles == 1 and torch.equal(one.coords, ref.coords) and torch.equal(one.feats, ref.feats)
    many = run_tex(16)
    assert tex.last_tiles > 8
    assert torch.equal(many.coords, ref.coords) and (many.feats - ref.feats).abs().max().item() <= 1e-4


def test_progress_line_per_tile(sp_env):
    sp, m = sp_env
    from trellis3 import decoder_tiling
    dec = make_decoder(m)
    lines = []
    decoder_tiling.install(dec, 16, log=lines.append)
    with torch.no_grad():
        dec(make_latent(sp), return_subs=True)
    assert len(lines) == dec.last_tiles > 8
    assert lines[0].startswith("  [shape tile 1/") and f"tile {dec.last_tiles}/{dec.last_tiles}]" in lines[-1]


def test_margin_is_computed_from_the_convolutions(sp_env):
    sp, m = sp_env
    from trellis3 import decoder_tiling
    dec = make_decoder(m, (1, 1, 1, 4, 0))
    # stage 3: 4 ConvNeXt convs (1 voxel each) + upsample conv1 (1) + conv2 after the upsample (half a voxel) = 5.5 -> 6
    assert decoder_tiling.required_margin(dec, 3) == 6


def test_setting_reads_env_and_explicit_beats_it(monkeypatch):
    from trellis3 import decoder_tiling
    monkeypatch.delenv("TRELLIS3_TILE_VOXELS", raising=False)
    assert decoder_tiling.tile_voxels_setting() == 0
    monkeypatch.setenv("TRELLIS3_TILE_VOXELS", "256")
    assert decoder_tiling.tile_voxels_setting() == 256
    assert decoder_tiling.tile_voxels_setting(0) == 0
    monkeypatch.setenv("TRELLIS3_TILE_VOXELS", "nonsense")
    assert decoder_tiling.tile_voxels_setting() == 0
