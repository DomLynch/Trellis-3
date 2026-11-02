"""Before/after check of the issue-188 memory work on saved latents: peak GPU memory, vertex/face counts and a hash of the positions.

    python tools/memcheck_188.py LATENTS.latents.pt --mode baseline            upstream decode (nothing released early), upstream to_glb
    python tools/memcheck_188.py LATENTS.latents.pt --mode lean [--tile N]     lean decode, attribute volume late, optional tiled decoder
    add --export-texture-size N to run export_mesh too (peak and GLB position hash are recorded for the export as well)

One mode per process (the peak counters start clean). Needs the GPU free and AC power; writes only under --out (default
<latents folder>/memcheck_188/). Prints one JSON line. Acceptance: vertex/face counts and `positions_sha256` equal between baseline and lean
(and with --tile 0 vs a single huge tile); a real tile size is allowed to differ by float noise (see trellis3.decoder_tiling).
"""
import argparse
import hashlib
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from trellis3.runner import bootstrap  # noqa: E402

bootstrap()   # must precede any torch import


def sha(arr):
    import numpy as np
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("latents")
    ap.add_argument("--mode", choices=["baseline", "lean"], required=True)
    ap.add_argument("--tile", type=int, default=0, help="tile_voxels for --mode lean (0 = off)")
    ap.add_argument("--export-texture-size", type=int, default=0, help="also run export_mesh at this texture size (0 = decode only)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--allow-battery", action="store_true")
    a = ap.parse_args()

    from trellis3 import power
    msg = power.refuse_on_battery(allow=a.allow_battery)
    if msg:
        print(msg)
        sys.exit(5)
    import torch
    if not torch.cuda.is_available():
        sys.exit("needs the GPU")
    free, total = torch.cuda.mem_get_info()
    if free < 0.6 * total:
        print(f"GPU busy: {free / 2**30:.1f} of {total / 2**30:.1f} GiB free; one GPU job at a time. Ask the lane that owns it for a gap.")
        sys.exit(6)

    if a.mode == "baseline":
        os.environ["TRELLIS3_GLB_LEAN"] = "0"
    os.environ["TRELLIS3_TILE_VOXELS"] = str(a.tile if a.mode == "lean" else 0)

    from trellis3.export import export_mesh
    from trellis3.latents import load_state
    from trellis3.runner import Runner
    from trellis3.stages import decode_state

    state = load_state(a.latents)
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(a.latents)), "memcheck_188")
    os.makedirs(out, exist_ok=True)
    runner = Runner(state["pipeline_type"], decode_only=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    mesh = decode_state(runner.pipeline, state, log=runner.log, lean=(a.mode == "lean"), attrs_to_cpu=(a.mode == "lean"))
    torch.cuda.synchronize()
    res = {"mode": a.mode, "tile": a.tile if a.mode == "lean" else 0, "latents": os.path.abspath(a.latents), "res": state.get("res"),
           "decode_s": round(time.time() - t0, 1), "decode_peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
           "vertices": int(mesh.vertices.shape[0]), "faces": int(mesh.faces.shape[0]),
           "positions_sha256": sha(mesh.vertices.detach().float().cpu().numpy()), "faces_sha256": sha(mesh.faces.detach().cpu().numpy()),
           "attrs_device": str(mesh.attrs.device)}
    if a.export_texture_size:
        torch.cuda.reset_peak_memory_stats()
        t1 = time.time()
        paths = export_mesh(mesh, os.path.join(out, f"{a.mode}_t{a.tile}"), texture_size=a.export_texture_size, log=runner.log)
        torch.cuda.synchronize()
        import trimesh
        g = trimesh.load(paths["glb"], force="mesh")
        res.update(export_s=round(time.time() - t1, 1), export_peak_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2),
                   glb_vertices=int(len(g.vertices)), glb_faces=int(len(g.faces)), glb_positions_sha256=sha(g.vertices.astype("float32")),
                   glb_lean=paths.get("glb_lean"))
    print(json.dumps(res))
    with open(os.path.join(out, f"{a.mode}_t{res['tile']}.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
