"""
Lean variant of `o_voxel.postprocess.to_glb` (export-side memory, TRELLIS.2 issue 188).

Reported in the issue (KaiStarkk comment, 2026-10-14): `to_glb` should keep the attribute volume off the GPU until the bake, and release the
previous `CuMesh` before building the remeshed one, plus chunk the narrow-band remesh; the three changes leave the output bit-identical.
Re-implemented here from that description and from reading the installed `to_glb`; no code was taken from the patch set the comment links to.

How: the installed `to_glb` is upstream Python, so the variant is made from its own source with two small text edits and then compiled in the
module's namespace. Edits that do not find their anchor are skipped and reported (a newer upstream just falls back to its own `to_glb`).

  attr_volume  `coords`, `aabb`, `voxel_size` and the attribute volume are moved to the GPU right before the attribute sampling (the bake), not
               before the remesh/simplify/UV steps, so the volume (millions of voxels) does not sit on the GPU while the mesh steps peak.
  old CuMesh   the remesh branch releases the old CuMesh (and empties the cache) before `remesh_narrow_band_dc` allocates its grid.
  chunking     `cumesh.remeshing.remesh_narrow_band_dc` of the installed cumesh already chunks its distance queries and voxel expansion; nothing to do.
"""

from __future__ import annotations

import re

RULES = []   # (name, anchor regex, replacement)

RULES.append((
    "attr_volume_to_gpu_at_the_bake",
    r"(\n    attrs = torch\.zeros\(texture_size, texture_size, attr_volume\.shape\[1\], device='cuda'\))",
    "\n    attr_volume, coords = attr_volume.cuda(), coords.cuda()   # trellis3.glb_lean: the volume reaches the GPU only for the bake\n"
    "    aabb, voxel_size = aabb.cuda(), voxel_size.cuda()\\1",
))
RULES.append((
    "remesh_center_on_gpu",
    r"(\n        center = aabb\.mean\(dim=0\))",
    "\\1.cuda()   # trellis3.glb_lean: aabb may still be on the CPU here",
))
RULES.append((
    "release_old_cumesh_before_the_remesh",
    r"(\n        mesh\.init\(\*cumesh\.remeshing\.remesh_narrow_band_dc\()",
    "\n        import gc as _gc\n        mesh = None   # trellis3.glb_lean: release the old CuMesh before the remesh allocates its grid\n"
    "        _gc.collect()\n        torch.cuda.empty_cache()\n        mesh = cumesh.CuMesh()\\1",
))


def patched_source(src: str):
    """(new source, applied rule names, skipped rule names)."""
    applied, skipped = [], []
    for name, pat, rep in RULES:
        new, n = re.subn(pat, rep, src, count=1)
        if n:
            src = new
            applied.append(name)
        else:
            skipped.append(name)
    return src, applied, skipped


def get_to_glb():
    """(to_glb function, volume_reaches_gpu_late, applied rules, skipped rules). The patched variant when its main rule applied, else upstream's own function."""
    import inspect
    import os
    import o_voxel.postprocess as pp
    if os.environ.get("TRELLIS3_GLB_LEAN") == "0":   # off switch for before/after measurements (tools/memcheck_188.py --mode baseline)
        return pp.to_glb, False, [], [r[0] for r in RULES]
    try:
        src, applied, skipped = patched_source(inspect.getsource(pp.to_glb))
        if "attr_volume_to_gpu_at_the_bake" not in applied:
            raise RuntimeError("anchor not found")
        ns = dict(pp.__dict__)
        exec(compile(src, "<trellis3.glb_lean to_glb>", "exec"), ns)
        return ns["to_glb"], True, applied, skipped
    except Exception:
        return pp.to_glb, False, [], [r[0] for r in RULES]
