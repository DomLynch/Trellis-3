"""
Exact trilinear sampling of sparse voxel attributes — the maths of the
official TRELLIS.2 bake, on CPU.

Microsoft's `to_glb` samples the PBR voxel volume at every texel with
`grid_sample_3d(..., mode='trilinear')` on index coordinates
`(pos - aabb[0]) / voxel_size` (align_corners semantics: voxel `c` sits at
index `c`). The trellis-mac CPU fallback replaced that with k-nearest-
neighbour inverse-distance averaging, which blurs. This module reproduces
the official trilinear sample over the *sparse* grid with a sorted-key
lookup, chunked so an 8K bake (67M texels) stays within a few GB.

One deliberate difference: a corner that has no voxel contributes nothing
and the remaining weights are renormalised. The official code multiplies
missing corners by zero without renormalising, which darkens texels at the
edge of the voxel shell; renormalising keeps the colour of the voxels that
exist. `renormalize=False` gives the official behaviour exactly.
"""

import numpy as np

_CORNERS = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)], dtype=np.int64)


def sparse_trilinear(coords, attrs, origin, voxel_size, points, renormalize=True, chunk=1_000_000):
    """Sample `attrs` [N,C] stored at integer `coords` [N,3] at world `points` [M,3].

    Returns (values [M,C] float32, covered [M] bool — at least one corner hit).
    """
    import torch
    from .voxel_lookup import CoordIndex

    coords = np.asarray(coords, dtype=np.int64)
    attrs = np.asarray(attrs, dtype=np.float32)
    origin = np.asarray(origin, dtype=np.float32).reshape(3)
    vs = np.asarray(voxel_size, dtype=np.float32).reshape(-1)
    vs = np.broadcast_to(vs, (3,)) if vs.size == 1 else vs
    points = np.asarray(points, dtype=np.float32)
    M, C = len(points), attrs.shape[1]
    index = CoordIndex(torch.from_numpy(coords))
    out = np.zeros((M, C), dtype=np.float32)
    covered = np.zeros(M, dtype=bool)
    for s in range(0, M, chunk):
        p = points[s:s + chunk]
        g = (p - origin) / vs                              # index space, voxel c at index c
        i0 = np.floor(g).astype(np.int64)
        f = (g - i0).astype(np.float32)                    # [m,3] in [0,1)
        acc = np.zeros((len(p), C), dtype=np.float32)
        wsum = np.zeros((len(p), 1), dtype=np.float32)
        for corner in _CORNERS:
            w = np.prod(np.where(corner == 1, f, 1.0 - f), axis=1)[:, None]   # [m,1]
            idx = index.lookup(torch.from_numpy(i0 + corner)).numpy()
            hit = idx >= 0
            if not hit.any():
                continue
            acc[hit] += w[hit] * attrs[idx[hit]]
            wsum[hit] += w[hit]
        has = wsum[:, 0] > 0
        if renormalize:
            acc[has] /= wsum[has]
        out[s:s + chunk] = acc
        covered[s:s + chunk] = has
    return out, covered
