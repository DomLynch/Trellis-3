"""
Post-processing applied to what the Metal baker returns, before writing GLB.

The Metal path (`o_voxel.postprocess.to_glb`, Pedro Naugusto's fork) differs
from the KDTree fallback in two ways that showed up as "wrong" in real runs:

- it exports `metallicFactor=1.0` multiplied by the model's metallic texture,
  while the KDTree path uses 0.0. Model metallic drifts high on bf16/MPS, so
  skin came out as chrome. `set_metallic_factor` lets the caller pin it.
- it writes base colour linearly with no gamma, while the KDTree path applies
  1/2.2. Same asset looks much darker through Metal. `apply_gamma` adds an
  optional curve so both paths can be matched to the input photo.

`fix_winding` runs before the bake on the simplified mesh: a consistent
winding pass plus outward-orientation check (trimesh.repair.fix_normals).
Upstream relies on the Metal cumesh `unify_face_orientations`, which left
~half the faces flipped in practice.
"""

import numpy as np


def _materials(obj):
    """Yield PBR materials from a trimesh.Trimesh or trimesh.Scene."""
    geoms = getattr(obj, "geometry", None)
    items = list(geoms.values()) if isinstance(geoms, dict) else [obj]
    for g in items:
        vis = getattr(g, "visual", None)
        mat = getattr(vis, "material", None)
        if mat is not None:
            yield mat


def set_metallic_factor(obj, factor):
    """Pin metallicFactor on every PBR material. Returns count changed."""
    if factor is None:
        return 0
    n = 0
    for mat in _materials(obj):
        if hasattr(mat, "metallicFactor"):
            mat.metallicFactor = float(factor)
            n += 1
    return n


def apply_gamma(obj, gamma):
    """Apply out = in ** (1/gamma) to each baseColorTexture's RGB (alpha kept)."""
    if not gamma or gamma == 1.0:
        return 0
    from PIL import Image
    n = 0
    for mat in _materials(obj):
        tex = getattr(mat, "baseColorTexture", None)
        if tex is None:
            continue
        arr = np.asarray(tex.convert("RGBA")).astype(np.float32) / 255.0
        arr[..., :3] = np.power(np.clip(arr[..., :3], 0, 1), 1.0 / gamma)
        mat.baseColorTexture = Image.fromarray((arr * 255 + 0.5).astype(np.uint8), "RGBA")
        n += 1
    return n


def fix_winding(vertices, faces):
    """Return (vertices, faces, n_flipped) with consistent, outward winding."""
    import trimesh
    m = trimesh.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(faces), process=False)
    before = np.asarray(m.faces).copy()
    trimesh.repair.fix_normals(m, multibody=True)
    after = np.asarray(m.faces)
    flipped = int((before != after).any(axis=1).sum())
    return np.asarray(m.vertices), after, flipped
