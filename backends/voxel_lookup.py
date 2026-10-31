"""
Vectorised integer-coordinate lookup, replacing Python dicts keyed by tuples.

Upstream trellis-mac builds `coord_to_idx = {tuple: i}` over every active
voxel and then loops `.item()` calls per query. At 512 that is ~800K entries
(seconds, ~0.5 GB of Python objects); at 1024 it is several million
(minutes, multiple GB). This packs each coordinate into one int64 mixed-radix
key, sorts once, and answers every query with one searchsorted.

The patch script copies this file into the TRELLIS.2 tree (as
t3_voxel_lookup.py) next to the modules that use it, so it must stay
dependency-free beyond torch.
"""

import torch


class CoordIndex:
    """Map integer coordinates [N, D] to their row index; -1 for misses."""

    def __init__(self, coords):
        c = coords.reshape(-1, coords.shape[-1]).to(torch.int64)
        self.D = c.shape[1]
        if c.shape[0] == 0:
            self.mins = torch.zeros(self.D, dtype=torch.int64, device=c.device)
            self.extent = torch.ones(self.D, dtype=torch.int64, device=c.device)
        else:
            self.mins = c.min(dim=0).values - 1
            self.extent = c.max(dim=0).values - self.mins + 2   # rel coords in [1, extent-2]
        self.strides = torch.ones(self.D, dtype=torch.int64, device=c.device)
        for d in range(self.D - 2, -1, -1):
            self.strides[d] = self.strides[d + 1] * self.extent[d + 1]
        keys = self._pack(c - self.mins)
        self.sorted_keys, self.order = torch.sort(keys)

    def _pack(self, rel):
        return (rel * self.strides).sum(dim=1)

    def lookup(self, queries):
        """queries [..., D] int -> [...] int64 row index, or -1 when absent."""
        shape = queries.shape[:-1]
        rel = queries.reshape(-1, self.D).to(torch.int64) - self.mins
        inside = (rel >= 0).all(dim=1) & (rel < self.extent).all(dim=1)
        keys = self._pack(rel.clamp(torch.zeros_like(self.extent), self.extent - 1))
        n = self.sorted_keys.numel()
        if n == 0:
            return torch.full(shape, -1, dtype=torch.int64, device=queries.device)
        pos = torch.searchsorted(self.sorted_keys, keys).clamp(max=n - 1)
        hit = inside & (self.sorted_keys[pos] == keys)
        return torch.where(hit, self.order[pos], torch.full_like(pos, -1)).reshape(shape)
