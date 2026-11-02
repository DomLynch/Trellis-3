"""export._bake_normals feeds the GLB's own vertex normals (axis-remapped like the positions) into the baker, so the map is encoded in the frame the renderer rebuilds."""
import numpy as np
import trimesh

import trellis3.export as ex
import trellis3.hires as hr


def test_vertex_normals_reach_the_baker_in_the_decoder_frame(monkeypatch):
    seen = {}

    def fake_bake(V, F, uv, hires, size, v_down=False, log=print, low_normals=None, **kw):
        seen.update(V=V, normals=low_normals, v_down=v_down)
        return np.full((size, size, 3), 128, np.uint8), np.ones((size, size), bool)

    monkeypatch.setattr(hr, "bake_normal_map", fake_bake)
    monkeypatch.setattr(hr, "attach_normal_map", lambda obj, img: 1)
    ico = trimesh.creation.icosphere(subdivisions=1)
    glb = trimesh.Trimesh(ico.vertices, ico.faces, vertex_normals=ico.vertex_normals, process=False)
    uvs = np.zeros((len(glb.vertices), 2))
    ex._bake_normals(glb, uvs, object(), 8, v_down=False, flip_v=False, in_glb_frame=True, log=lambda *_: None)
    assert np.allclose(seen["normals"], hr.glb_to_decoder_frame(np.asarray(glb.vertex_normals, np.float32)), atol=1e-6)
    assert np.allclose(seen["V"], hr.glb_to_decoder_frame(np.asarray(glb.vertices)))   # positions and normals share one axis remap
    ex._bake_normals(glb, uvs, object(), 8, v_down=True, flip_v=False, in_glb_frame=False, log=lambda *_: None)
    assert np.allclose(seen["normals"], np.asarray(glb.vertex_normals, np.float32), atol=1e-6)   # CPU-path frame: no remap


def test_peak_gb_uses_gibibytes(monkeypatch):
    import types
    import torch
    import trellis3.stages as st
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 2 ** 30 * 20)
    assert st._peak_gb() == 20.0
