"""Export tweaks of the issue-188 memory work: trellis3.glb_lean is an owned copy of o_voxel's to_glb with two marked changes. CPU only."""
import importlib.util
import inspect
import os
import re

import pytest

from trellis3 import glb_lean


def _own_source():
    return inspect.getsource(glb_lean.to_glb_lean)


def test_the_two_changes_are_in_the_right_order():
    src = _own_source()
    assert src.index("attr_volume, coords = attr_volume.cuda(), coords.cuda()") < src.index("attrs = torch.zeros(")   # volume reaches the GPU right before the bake
    assert src.index("mesh = None") < src.index("remesh_narrow_band_dc(")                                             # old CuMesh released before the remesh
    assert "aabb.mean(dim=0).cuda()" in src
    assert glb_lean.get_to_glb() == (glb_lean.to_glb_lean, True)


def test_off_switch_returns_upstream(monkeypatch):
    if importlib.util.find_spec("o_voxel") is None:
        pytest.skip("o_voxel not installed here")
    monkeypatch.setenv("TRELLIS3_GLB_LEAN", "0")
    import types, sys
    fake = types.ModuleType("o_voxel.postprocess")
    fake.to_glb = lambda: "upstream"
    monkeypatch.setitem(sys.modules, "o_voxel.postprocess", fake)
    fn, late = glb_lean.get_to_glb()
    assert fn() == "upstream" and late is False


def _normalise(src):
    src = re.sub(r"def to_glb(_lean)?\(", "def to_glb(", src)
    src = re.sub(r"\n    import gc\n.*?    import cumesh\n", "\n", src, flags=re.S)           # heavy imports moved inside the copy
    src = src.replace("        center = aabb.mean(dim=0).cuda()   # trellis3: aabb may still be on the CPU here\n", "        center = aabb.mean(dim=0)\n")
    src = re.sub(r"    # trellis3 change 1:.*?\n    attr_volume, coords = attr_volume.cuda\(\), coords.cuda\(\)\n    aabb, voxel_size = aabb.cuda\(\), voxel_size.cuda\(\)\n", "", src)
    src = re.sub(r"        # trellis3 change 2:.*?\n        mesh = None\n        gc.collect\(\)\n        torch.cuda.empty_cache\(\)\n        mesh = cumesh.CuMesh\(\)\n", "", src)
    return "\n".join(line.rstrip() for line in src.splitlines() if line.strip())


def test_copy_equals_the_installed_function_apart_from_the_marked_changes():
    """Fails when upstream's to_glb changes, so the owned copy is updated deliberately."""
    spec = importlib.util.find_spec("o_voxel")
    if spec is None or not spec.origin:
        pytest.skip("o_voxel not installed here")
    text = open(os.path.join(os.path.dirname(spec.origin), "postprocess.py"), encoding="utf-8").read().replace("\r\n", "\n")
    upstream = re.search(r"\ndef to_glb\(.*", text, re.S).group(0)
    assert _normalise(_own_source()) == _normalise(upstream)
