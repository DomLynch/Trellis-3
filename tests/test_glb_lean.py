"""Export tweaks of the issue-188 memory work: the source edits of trellis3.glb_lean on the installed o_voxel to_glb and on a synthetic source. CPU only."""
import importlib.util
import os
import re

import pytest

from trellis3 import glb_lean

SYNTH = '''
def to_glb(vertices, faces, attr_volume, coords, aabb, voxel_size):
    mesh = cumesh.CuMesh()
    if not remesh:
        pass
    else:
        center = aabb.mean(dim=0)
        mesh.init(*cumesh.remeshing.remesh_narrow_band_dc(
            vertices, faces,
        ))
    attrs = torch.zeros(texture_size, texture_size, attr_volume.shape[1], device='cuda')
    return attrs
'''


def test_all_three_edits_apply_to_the_expected_shape_and_compile():
    src, applied, skipped = glb_lean.patched_source(SYNTH)
    assert applied == [r[0] for r in glb_lean.RULES] and skipped == []
    assert src.index("attr_volume, coords = attr_volume.cuda(), coords.cuda()") < src.index("attrs = torch.zeros")   # moved right before the bake
    assert src.index("mesh = None") < src.index("remesh_narrow_band_dc")                                         # old CuMesh released first
    assert "aabb.mean(dim=0).cuda()" in src
    compile(src, "<x>", "exec")


def test_an_anchor_that_is_gone_is_skipped_not_forced():
    src, applied, skipped = glb_lean.patched_source("def to_glb():\n    return 1\n")
    assert applied == [] and len(skipped) == 3 and src.endswith("return 1\n")


def test_installed_upstream_source_still_matches_if_present():
    spec = importlib.util.find_spec("o_voxel")
    if spec is None or not spec.origin:
        pytest.skip("o_voxel not installed here")
    path = os.path.join(os.path.dirname(spec.origin), "postprocess.py")
    text = open(path, encoding="utf-8").read()
    m = re.search(r"\ndef to_glb\(.*?(?=\n\ndef |\Z)", text, re.S)
    src, applied, skipped = glb_lean.patched_source(m.group(0))
    assert "attr_volume_to_gpu_at_the_bake" in applied, skipped
    compile(src, "<installed to_glb>", "exec")
