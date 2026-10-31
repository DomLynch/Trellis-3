import time

import numpy as np
import torch

from backends.voxel_lookup import CoordIndex


def _ref(coords, queries):
    d = {tuple(c): i for i, c in enumerate(coords.tolist())}
    return torch.tensor([d.get(tuple(q), -1) for q in queries.tolist()], dtype=torch.int64)


def test_lookup_matches_dict_reference():
    rng = np.random.default_rng(0)
    coords = torch.tensor(np.unique(rng.integers(-5, 40, size=(3000, 3)), axis=0))
    queries = torch.tensor(rng.integers(-8, 44, size=(5000, 3)))
    idx = CoordIndex(coords)
    got = idx.lookup(queries)
    assert torch.equal(got, _ref(coords, queries))
    # every present coordinate maps to itself
    assert torch.equal(idx.lookup(coords), torch.arange(len(coords)))


def test_lookup_shapes_and_out_of_range():
    coords = torch.tensor([[0, 0, 0], [1, 2, 3]])
    idx = CoordIndex(coords)
    q = torch.tensor([[[0, 0, 0], [1, 2, 3]], [[9, 9, 9], [-1, 0, 0]]])
    got = idx.lookup(q)
    assert got.shape == (2, 2)
    assert got.tolist() == [[0, 1], [-1, -1]]
    big = torch.tensor([[10 ** 6, -10 ** 6, 5]])
    assert idx.lookup(big).item() == -1


def test_four_dim_coords_with_batch():
    coords = torch.tensor([[0, 1, 1, 1], [1, 1, 1, 1], [0, 2, 1, 1]])
    idx = CoordIndex(coords)
    assert idx.lookup(torch.tensor([[1, 1, 1, 1], [0, 2, 1, 1], [1, 2, 1, 1]])).tolist() == [1, 2, -1]


def test_empty_index():
    idx = CoordIndex(torch.zeros((0, 3), dtype=torch.int64))
    assert idx.lookup(torch.tensor([[1, 2, 3]])).tolist() == [-1]


def test_scales_to_decoder_size():
    rng = np.random.default_rng(1)
    coords = torch.tensor(np.unique(rng.integers(0, 1024, size=(2_000_000, 3)), axis=0))
    t = time.time()
    idx = CoordIndex(coords)
    got = idx.lookup(coords + torch.tensor([0, 0, 1]))
    dt = time.time() - t
    assert dt < 20, f"too slow: {dt:.1f}s"
    assert (got >= 0).sum() > 0
