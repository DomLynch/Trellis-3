import itertools

import numpy as np
import torch

from backends.conv_none import build_neighbor_map


def _reference(coords, ks, dil):
    Kd, Kh, Kw = ks
    dz, dy, dx = dil
    d = {tuple(c): i for i, c in enumerate(coords.tolist())}
    edges = set()
    for kz, ky, kx in itertools.product(range(Kd), range(Kh), range(Kw)):
        oz, oy, ox = (kz - Kd // 2) * dz, (ky - Kh // 2) * dy, (kx - Kw // 2) * dx
        kidx = kz * Kh * Kw + ky * Kw + kx
        for i, (b, z, y, x) in enumerate(coords.tolist()):
            j = d.get((b, z + oz, y + oy, x + ox))
            if j is not None:
                edges.add((j, i, kidx))
    return edges


def test_neighbor_map_matches_dict_reference():
    rng = np.random.default_rng(0)
    c = np.unique(rng.integers(0, 6, size=(300, 3)), axis=0)
    coords = torch.tensor(np.concatenate([np.zeros((len(c), 1), int), c], axis=1))
    coords = torch.cat([coords, coords[:40] + torch.tensor([1, 0, 0, 0])])  # second batch item
    for ks, dil in (((3, 3, 3), (1, 1, 1)), ((3, 3, 3), (2, 2, 2)), ((1, 3, 3), (1, 1, 1))):
        src, tgt, kid = build_neighbor_map(coords, ks, dil, torch.device("cpu"))
        got = set(zip(src.tolist(), tgt.tolist(), kid.tolist()))
        assert got == _reference(coords, ks, dil), (ks, dil)


def test_neighbor_map_empty():
    src, tgt, kid = build_neighbor_map(torch.zeros((0, 4), dtype=torch.int64), (3, 3, 3), (1, 1, 1),
                                       torch.device("cpu"))
    assert src.numel() == tgt.numel() == kid.numel() == 0
