"""Patch 1 of the TRELLIS.2 issue 188 memory work: decode_state hands ownership of the latents to the lean decode, so they are really freed before
hole filling, and the attribute volume stays on the CPU until the bake. CPU-only, fake pipeline."""
import gc
import sys
import types
import weakref

import torch

from trellis3 import stages


class _Slat:
    def __init__(self, feats, coords):
        self.feats, self.coords = feats, coords

    def replace(self, feats):
        return _Slat(feats, self.coords)


class _Mesh:
    vertices = torch.zeros((3, 3))
    faces = torch.tensor([[0, 1, 2]])

    def __init__(self, probe):
        self.probe = probe

    def fill_holes(self):
        self.probe("fill_holes")


class _Vox:
    coords = torch.zeros((2, 4), dtype=torch.int32)
    feats = torch.zeros((2, 6))
    shape = torch.Size([2, 6])
    spatial_shape = torch.Size([4, 4, 4])


class _MWV:
    def __init__(self, vertices, faces, **kw):
        self.vertices, self.faces, self.kw = vertices, faces, kw


def _pipe(events):
    refs = {}

    class P:
        device = torch.device("cpu")
        pbr_attr_layout = {"base_color": slice(0, 3)}

        def decode_shape_slat(self, slat, res):
            refs["shape"] = weakref.ref(slat)
            return [_Mesh(lambda e: events.append((e, refs["shape"](), refs["tex"]())))], ["subs"]

        def decode_tex_slat(self, slat, subs):
            refs["tex"] = weakref.ref(slat)
            return [_Vox()]

    return P()


def _state():
    return {"shape_feats": torch.zeros((2, 32)), "shape_coords": torch.zeros((2, 4), dtype=torch.int32), "tex_feats": torch.zeros((2, 32)), "res": 512}


def test_latents_are_unreferenced_when_holes_are_filled(monkeypatch):
    reps = types.ModuleType("trellis2.representations")
    reps.MeshWithVoxel = _MWV
    monkeypatch.setitem(sys.modules, "trellis2.representations", reps)
    monkeypatch.setattr(stages, "_make_sparse", lambda f, c, d: _Slat(f, c))
    events = []
    out = stages.decode_state(_pipe(events), _state(), log=lambda *_: None)
    gc.collect()
    assert events == [("fill_holes", None, None)]   # neither the shape nor the texture latent is alive when fill_holes runs
    assert out.kw["attrs"].device.type == "cpu" and out.kw["coords"].shape == (2, 3)   # attribute volume kept off the device


def test_decode_runs_without_autograd(monkeypatch):
    """The root cause of the 1024/1536 decode OOMs: the decoders were called with grad enabled, so their activations were all kept for a backward pass."""
    reps = types.ModuleType("trellis2.representations")
    reps.MeshWithVoxel = _MWV
    monkeypatch.setitem(sys.modules, "trellis2.representations", reps)
    seen = {}

    class P(_pipe([]).__class__):
        def decode_shape_slat(self, slat, res):
            seen["grad"] = torch.is_grad_enabled()
            return super().decode_shape_slat(slat, res)

    stages._decode_lean(P(), [_Slat(None, None), _Slat(None, None)], 512)
    assert seen["grad"] is False


def test_offload_state_moves_only_device_tensors_and_keeps_values():
    st = {"a": torch.arange(3), "n": 5, "s": "x"}
    assert stages._offload_state(st) == []   # CPU tensors stay
    assert st["a"].tolist() == [0, 1, 2] and st["n"] == 5


def test_wrapper_keeps_the_old_signature_and_output(monkeypatch):
    reps = types.ModuleType("trellis2.representations")
    reps.MeshWithVoxel = _MWV
    monkeypatch.setitem(sys.modules, "trellis2.representations", reps)
    events = []
    out = stages.decode_latent_lean(_pipe(events), _Slat(None, None), _Slat(None, None), 1024, attrs_to_cpu=False)
    assert out[0].kw["voxel_size"] == 1 / 1024 and out[0].kw["origin"] == [-0.5, -0.5, -0.5]
    assert out[0].kw["attrs"].shape == (2, 6)
