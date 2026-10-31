"""
CPU replacements for the cumesh operations TRELLIS.2 runs at decode time.

Upstream trellis-mac stubs Mesh.fill_holes / remove_faces / simplify to a bare
`return` because the Metal port of cumesh segfaults on decoder-sized meshes
(400K+ vertices). That leaves small holes in every exported mesh.

This module reimplements the three ops in numpy so they run on any device.
The hole filler matches cumesh semantics: only boundary loops whose perimeter
is below `max_hole_perimeter` are closed (meshes are in [-0.5, 0.5]^3, so the
default 3e-2 only touches tiny gaps and never seals an intentional opening).

The patch script copies this file to
    TRELLIS.2/trellis2/representations/mesh/cleanup_cpu.py
and rewrites Mesh.fill_holes / remove_faces / simplify to call into it.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Boundary loops
# ---------------------------------------------------------------------------

def boundary_loops(faces):
    """Return a list of boundary loops, each an ordered array of vertex ids.

    A boundary edge is one referenced by exactly one triangle. Loops are
    traced by walking boundary edges in the triangle's winding direction so
    the resulting loop orientation is consistent with the surface normal.
    Non-manifold junctions (a boundary vertex with more than two boundary
    edges) terminate a walk; the partial loop is dropped rather than guessed.
    """
    faces = np.asarray(faces, dtype=np.int64)
    if faces.size == 0:
        return []

    # Directed edges in winding order: (a,b), (b,c), (c,a)
    directed = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
    # Undirected key for counting
    lo = directed.min(axis=1)
    hi = directed.max(axis=1)
    key = lo * (faces.max() + 1) + hi
    _, inv, counts = np.unique(key, return_inverse=True, return_counts=True)
    is_boundary = counts[inv] == 1
    b_edges = directed[is_boundary]
    if b_edges.size == 0:
        return []

    # next-vertex map along boundary; detect non-manifold boundary vertices
    nxt = {}
    bad = set()
    for a, b in b_edges:
        a = int(a); b = int(b)
        if a in nxt:
            bad.add(a)
        nxt[a] = b

    loops = []
    visited = set()
    for start in list(nxt.keys()):
        if start in visited or start in bad:
            continue
        loop = [start]
        visited.add(start)
        cur = nxt[start]
        ok = True
        while cur != start:
            if cur in visited or cur in bad or cur not in nxt:
                ok = False
                break
            loop.append(cur)
            visited.add(cur)
            cur = nxt[cur]
        if ok and len(loop) >= 3:
            loops.append(np.asarray(loop, dtype=np.int64))
    return loops


def loop_perimeter(vertices, loop):
    pts = vertices[loop]
    nxt = np.roll(pts, -1, axis=0)
    return float(np.linalg.norm(nxt - pts, axis=1).sum())


# ---------------------------------------------------------------------------
# Public ops (numpy in, numpy out)
# ---------------------------------------------------------------------------

def fill_holes_np(vertices, faces, max_hole_perimeter=3e-2):
    """Close small boundary loops with a centroid fan.

    Returns (vertices, faces, n_filled). Loops longer than max_hole_perimeter
    are left open. Each filled loop adds one vertex (its centroid) and
    len(loop) triangles wound to match the surrounding surface.
    """
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int64)
    loops = boundary_loops(faces)
    if not loops:
        return vertices, faces, 0

    new_verts = []
    new_faces = []
    n_v = len(vertices)
    for loop in loops:
        if loop_perimeter(vertices, loop) > max_hole_perimeter:
            continue
        c = vertices[loop].mean(axis=0)
        ci = n_v + len(new_verts)
        new_verts.append(c)
        # boundary edge (a->b) is wound with the triangle; the hole is on the
        # other side, so the filling triangle is (b, a, centroid).
        a = loop
        b = np.roll(loop, -1)
        new_faces.append(np.stack([b, a, np.full_like(a, ci)], axis=1))

    if not new_verts:
        return vertices, faces, 0
    vertices = np.concatenate([vertices, np.asarray(new_verts, dtype=np.float32)], axis=0)
    faces = np.concatenate([faces] + new_faces, axis=0)
    return vertices, faces, len(new_verts)


def remove_faces_np(vertices, faces, face_mask):
    """Drop faces where face_mask is True and compact unreferenced vertices."""
    vertices = np.asarray(vertices)
    faces = np.asarray(faces, dtype=np.int64)
    face_mask = np.asarray(face_mask, dtype=bool)
    keep = faces[~face_mask]
    if keep.size == 0:
        return vertices[:0], keep.reshape(0, 3)
    used = np.unique(keep)
    remap = np.full(len(vertices), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    return vertices[used], remap[keep]


def simplify_np(vertices, faces, target=1000000):
    """Quadric decimation to roughly `target` faces via fast_simplification.

    No-op if the mesh already has <= target faces or the package is missing.
    """
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int64)
    if len(faces) <= target:
        return vertices, faces
    try:
        import fast_simplification
    except ImportError:
        return vertices, faces
    ratio = 1.0 - target / len(faces)
    v, f = fast_simplification.simplify(vertices, faces.astype(np.int32), ratio)
    return np.asarray(v, dtype=np.float32), np.asarray(f, dtype=np.int64)


# ---------------------------------------------------------------------------
# Torch wrappers used by the patched Mesh class
# ---------------------------------------------------------------------------

def _to_np(t):
    return t.detach().cpu().numpy()


def fill_holes_tensor(vertices, faces, max_hole_perimeter=3e-2):
    import torch
    v, f, n = fill_holes_np(_to_np(vertices), _to_np(faces), max_hole_perimeter)
    if n == 0:
        return vertices, faces
    return (torch.from_numpy(v).to(vertices.device, dtype=vertices.dtype),
            torch.from_numpy(f.astype(np.int32)).to(faces.device))


def remove_faces_tensor(vertices, faces, face_mask):
    import torch
    v, f = remove_faces_np(_to_np(vertices), _to_np(faces), _to_np(face_mask))
    return (torch.from_numpy(np.ascontiguousarray(v)).to(vertices.device, dtype=vertices.dtype),
            torch.from_numpy(f.astype(np.int32)).to(faces.device))


def simplify_tensor(vertices, faces, target=1000000):
    import torch
    v, f = simplify_np(_to_np(vertices), _to_np(faces), target)
    return (torch.from_numpy(np.ascontiguousarray(v)).to(vertices.device, dtype=vertices.dtype),
            torch.from_numpy(f.astype(np.int32)).to(faces.device))
