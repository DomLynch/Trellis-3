"""
Tiled run of the shape decoder's last two stages (peak GPU memory).

Why: at the 1536 cascade the decoder's last upsample block (`SparseResBlockC2S3d`, the 3x3x3 sparse convolution that
doubles the resolution) asks for more memory than a 16 GB card has. Every layer of the last two stages is a 3x3x3 sparse
convolution or acts per voxel, so a voxel's output depends only on its neighbourhood: the stages can run on spatial blocks
and the results be stitched. Reported in microsoft/TRELLIS.2 issue 188 (comment by KaiStarkk, 2026-10-14), re-implemented
here from that description and from reading the decoder; no code was taken from the patch set that comment links to.

What it does
------------
`tiled_forward` replaces `SparseUnetVaeDecoder.forward` for inference when `decoder.tile_voxels > 0`:

  * stages 0 .. n-3 run on the whole tensor, exactly as upstream;
  * the voxels entering stage n-2 are cut into cubes of `tile_voxels` per side (the "core"), each run with a margin of at
    least `margin` voxels of neighbours (8 by default, raised if the stages' convolutions reach further), through stage n-2 (with its
    upsample block), stage n-1, the final layer norm and the output layer;
  * only the rows that belong to the core are kept, and they are written back at their original position, so the
    output rows (and the subdivision tensor the texture decoder is guided by) come out in the same order as untiled.

A single tile that covers everything runs the same operations on the same rows and reproduces the untiled output exactly.
Smaller tiles can differ in the last bits (the sparse-convolution kernels pick their summation order per input size);
issue 188 measured a largest feature difference of 0.016 at 1024 with 128-voxel tiles.

Switch: `TRELLIS3_TILE_VOXELS` (0 = off, the default) or `install(decoder, tile_voxels)`. Training mode and decoders without
predicted subdivision use the upstream forward unchanged.
"""

from __future__ import annotations

import math
import os

DEFAULT_MARGIN = 8
_PATCHED = "_trellis3_tiled"


def tile_voxels_setting(explicit=None) -> int:
    """Voxels per tile side, 0 = off. `explicit` beats the environment."""
    if explicit is not None:
        return max(0, int(explicit))
    try:
        return max(0, int(os.environ.get("TRELLIS3_TILE_VOXELS", "0")))
    except ValueError:
        return 0


def conv_radius(module) -> float:
    """Reach, in voxels of the module's own resolution, of one sparse convolution (kernel // 2 * dilation)."""
    k = getattr(module, "kernel_size", None)
    d = getattr(module, "dilation", 1)
    try:
        k = max(k) if hasattr(k, "__iter__") else int(k)
        d = max(d) if hasattr(d, "__iter__") else int(d)
    except (TypeError, ValueError):
        return 1.0
    return float(k // 2 * d)


def required_margin(decoder, first: int) -> int:
    """Voxels (at the resolution entering stage `first`) a tile needs around its core so that its core output equals the untiled one.
    Counts the reach of every convolution from stage `first` on; convolutions after the upsample block count half per factor-2 step."""
    from trellis2.modules import sparse as sp
    reach, scale = 0.0, 1.0
    for i in range(first, len(decoder.blocks)):
        for j, block in enumerate(decoder.blocks[i]):
            for m in block.modules():
                if isinstance(m, sp.SparseConv3d):
                    # SparseResBlockC2S3d: conv1 runs before the upsample (at `scale`), conv2 after it (at scale / 2)
                    after = type(block).__name__ == "SparseResBlockC2S3d" and m is getattr(block, "conv2", None)
                    reach += conv_radius(m) * (scale / 2 if after else scale)
            if type(block).__name__ == "SparseResBlockC2S3d":
                scale /= 2
    return int(math.ceil(reach))


def _run_stage(decoder, i, h, subs):
    """One decoder stage, the same loop body as upstream forward (inference, predicted subdivision)."""
    res = decoder.blocks[i]
    for j, block in enumerate(res):
        if i < len(decoder.blocks) - 1 and j == len(res) - 1:
            h, sub = block(h)
            subs.append(sub)
        else:
            h = block(h)
    return h


def tiled_forward(decoder, x, guide_subs=None, return_subs=False):
    import torch
    import torch.nn.functional as F
    from trellis2.modules import sparse as sp

    T = int(decoder.tile_voxels)
    first = len(decoder.blocks) - 2
    h = decoder.from_latent(x)
    h = h.type(decoder.dtype)
    subs = []
    for i in range(first):
        h = _run_stage(decoder, i, h, subs)

    margin = max(int(getattr(decoder, "tile_margin", DEFAULT_MARGIN)), required_margin(decoder, first))
    coords, feats = h.coords, h.feats
    n = coords.shape[0]
    cell = torch.div(coords[:, 1:], T, rounding_mode="floor")
    keys = torch.unique(torch.cat([coords[:, :1], cell], dim=1), dim=0)   # one tile per (batch, cell)

    decoder.last_tiles = int(keys.shape[0])   # for the receipt and the tests
    kept = []            # per tile: (parent index, child feats, child coords) of the core rows
    sub_rows = []        # per tile: (parent index, subdivision feats)
    scale = None
    out_dtype = None
    for key in keys:
        b, c0 = key[0], key[1:]
        core = (coords[:, 0] == b) & (cell == c0).all(dim=1)
        lo, hi = c0 * T - margin, (c0 + 1) * T + margin
        ext = (coords[:, 0] == b) & ((coords[:, 1:] >= lo) & (coords[:, 1:] < hi)).all(dim=1)
        idx = ext.nonzero().squeeze(1)
        core_in = core[idx]
        ht = sp.SparseTensor(feats[idx], coords[idx])
        ht._scale = h._scale
        tsubs = []
        for i in range(first, len(decoder.blocks)):
            ht = _run_stage(decoder, i, ht, tsubs)
        sub = tsubs[-1]                                    # subdivision of the stage-`first` upsample, one row per entering voxel
        ht = ht.type(x.dtype)
        ht = ht.replace(F.layer_norm(ht.feats, ht.feats.shape[-1:]))
        ht = decoder.output_layer(ht)
        n_leaf = (sub.feats > 0).sum(dim=-1)
        parent = torch.repeat_interleave(torch.arange(idx.shape[0], device=idx.device), n_leaf, output_size=ht.coords.shape[0])
        rows = core_in[parent].nonzero().squeeze(1)
        kept.append((idx[parent[rows]], ht.feats[rows], ht.coords[rows]))
        sub_rows.append((idx[core_in.nonzero().squeeze(1)], sub.feats[core_in]))
        scale, out_dtype = ht._scale, ht.feats.dtype
        del ht, sub, tsubs, parent, rows, n_leaf
    del feats

    # --- stitch: the output of parent g sits at offsets[g] .. offsets[g] + n_leaf[g], exactly as untiled -----------------
    sub_full = torch.zeros((n, sub_rows[0][1].shape[1]), dtype=sub_rows[0][1].dtype, device=coords.device)
    for g, s in sub_rows:
        sub_full[g] = s
    n_leaf_full = (sub_full > 0).sum(dim=-1)
    offsets = torch.cumsum(n_leaf_full, dim=0) - n_leaf_full
    total = int(n_leaf_full.sum().item())
    out_feats = torch.empty((total, kept[0][1].shape[1]), dtype=out_dtype, device=coords.device)
    out_coords = torch.empty((total, 4), dtype=coords.dtype, device=coords.device)
    filled = 0
    for g, f, c in kept:
        if g.numel() == 0:
            continue
        _, counts = torch.unique_consecutive(g, return_counts=True)
        starts = torch.cumsum(counts, dim=0) - counts
        rank = torch.arange(g.shape[0], device=g.device) - torch.repeat_interleave(starts, counts)
        dest = offsets[g] + rank
        out_feats[dest], out_coords[dest] = f, c
        filled += g.shape[0]
    if filled != total:
        raise RuntimeError(f"tiled decode stitched {filled} of {total} rows; the margin or the tile size does not cover the stages")
    out = sp.SparseTensor(out_feats, out_coords)
    out._scale = scale
    sub_t = sp.SparseTensor(sub_full, coords)
    sub_t._scale = h._scale
    subs.append(sub_t)
    return (out, subs) if return_subs else out


def install(decoder, tile_voxels=None, margin=DEFAULT_MARGIN):
    """Switch tiling on for one decoder instance (0 = off). Patches the class forward once; returns the tile size in effect."""
    from trellis2.models.sc_vaes import sparse_unet_vae as m
    cls = m.SparseUnetVaeDecoder
    if not getattr(cls, _PATCHED, False):
        orig = cls.forward

        def forward(self, x, guide_subs=None, return_subs=False):
            if (getattr(self, "tile_voxels", 0) > 0 and not self.training and self.pred_subdiv and guide_subs is None
                    and len(self.blocks) >= 3):
                return tiled_forward(self, x, guide_subs, return_subs)
            return orig(self, x, guide_subs, return_subs)

        cls.forward = forward
        setattr(cls, _PATCHED, True)
    decoder.tile_voxels = tile_voxels_setting(tile_voxels)
    decoder.tile_margin = margin
    return decoder.tile_voxels
