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


def _mean_rgb(img, use_alpha=True):
    arr = np.asarray(img.convert("RGBA")).astype(np.float32) / 255.0
    if use_alpha and (arr[..., 3] < 0.999).any():
        w = arr[..., 3]
        sel = w > 0.5
        if sel.any():
            return arr[..., :3][sel].mean(axis=0)
    return arr[..., :3].reshape(-1, 3).mean(axis=0)


def auto_gamma(obj, reference, lo=1.0, hi=3.0):
    """Pick the gamma that makes the baked base colour's mean match `reference`.

    With out = in ** (1/g), matching means gives g = log(mean_tex) / log(mean_ref)
    (per channel, then averaged). The baked texture uses every texel (to_glb
    inpaints the atlas from the surface, so the mean is representative); the
    reference uses its alpha mask when it has one. Returns (gamma, mean_tex,
    mean_ref) or (None, ..) when it cannot be estimated.
    """
    from PIL import Image
    if isinstance(reference, str):
        reference = Image.open(reference)
    ref = _mean_rgb(reference)
    tex_means = [_mean_rgb(mat.baseColorTexture, use_alpha=False)
                 for mat in _materials(obj) if getattr(mat, "baseColorTexture", None) is not None]
    if not tex_means:
        return None, None, ref
    tex = np.mean(tex_means, axis=0)
    eps = 1e-3
    g = np.log(np.clip(tex, eps, 1 - eps)) / np.log(np.clip(ref, eps, 1 - eps))
    g = float(np.clip(np.mean(g), lo, hi))
    return g, tex, ref


def fix_winding(vertices, faces):
    """Return (vertices, faces, n_flipped) with consistent, outward winding."""
    import trimesh
    m = trimesh.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(faces), process=False)
    before = np.asarray(m.faces).copy()
    trimesh.repair.fix_normals(m, multibody=True)
    after = np.asarray(m.faces)
    flipped = int((before != after).any(axis=1).sum())
    return np.asarray(m.vertices), after, flipped
