"""Turn a saved Pixal3D decoded mesh into a GLB on the CPU only, in a fresh process (no GPU, no pipeline in memory).

    pixal3d_glb_from_decoded.py <decoded.pt> <out.glb> [--faces 1000000] [--tex 4096]
        --run-dir C:\\Users\\dom12\\Documents\\Pixal3D-run     (holds pixal3d_glb_cpu.py)

decoded.pt is written by tools/pixal3d_resume_tex.py (vertices, faces, coords, attrs, res). Running the CPU port here instead of inside the
inference process matters: the process that sampled the mesh still holds the pipeline and ~15 GB of GPU memory carved out of system RAM, and
the 15.8M-face clean/decimate pass ran the machine out of RAM there (guard kill at 0.1 GB free). The output matches inference_mv.py:
Y/Z rotation applied, webp extension.
"""
import argparse
import os
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("decoded")
    ap.add_argument("output")
    ap.add_argument("--faces", type=int, default=1_000_000)
    ap.add_argument("--tex", type=int, default=4096)
    ap.add_argument("--run-dir", default=r"C:\Users\dom12\Documents\Pixal3D-run")
    a = ap.parse_args()
    sys.path.insert(0, a.run_dir)
    import torch
    import pixal3d_glb_cpu as G

    d = torch.load(a.decoded, map_location="cpu")
    layout = {"base_color": slice(0, 3), "metallic": slice(3, 4), "roughness": slice(4, 5), "alpha": slice(5, 6)}
    print(f"[from_decoded] {d['vertices'].shape[0]:,} verts, {d['faces'].shape[0]:,} faces, {d['coords'].shape[0]:,} voxels, res {d['res']}",
          flush=True)
    glb = G.to_glb_cpu(d["vertices"], d["faces"], d["attrs"], d["coords"], layout, [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                       grid_size=int(d["res"]), decimation_target=a.faces, texture_size=a.tex)
    rot = np.array([[-1, 0, 0, 0], [0, 0, -1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=np.float64)
    glb.apply_transform(rot)
    glb.export(a.output, extension_webp=True)
    print(f"[Done] GLB saved to: {a.output}", flush=True)


if __name__ == "__main__":
    main()
