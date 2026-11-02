"""Landmark registration helper for the projection step (CPU).

  render  <mesh.glb> <outdir> [--yaws 0 90]     splat-render the mesh from exact orthographic cameras, with a pixel grid, for
                                                picking mesh landmarks by eye; writes <outdir>/render_yaw<N>.png, _grid.png and meta.json
  grid    <image.png> <out.png> [--step 100]    draw a labelled pixel grid on any image (for picking image landmarks)
  crop    <image.png> <out.png> x0 y0 x1 y1 zoom [step]   zoomed, gridded crop (labels are ORIGINAL image pixels) for picking
  solve   <landmarks.json> <outdir>             solve a camera per view from the picks, write cameras.json and overlay images

landmarks.json:
  {"mesh": {"yaw0": {"nose_tip": [px, py], ...}, "yaw90": {"ear_vr": [px, py]}},        # picks in the render_yaw<N> images
   "views": {"cut_front.png": {"yaw": 0, "path": "...", "picks": {"eye_vl": [x, y], ...}},
             "cut_tq_mirror.png": {"yaw": -45, "path": "...", "mirror_of": "cut_tq.png"}, ...}}
Labels: eye_vl / eye_vr (viewer-left / viewer-right eye in the FRONT view), nose_tip, mouth_vl, mouth_vr, chin, ear_vr, ear_vl.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trellis3 import projection as pj          # noqa: E402
from trellis3 import registration as reg       # noqa: E402


def _arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def draw_grid(img, step=100, label=True):
    from PIL import Image, ImageDraw
    im = Image.fromarray(img) if isinstance(img, np.ndarray) else img.copy()
    dr = ImageDraw.Draw(im)
    w, h = im.size
    for x in range(0, w, step):
        dr.line([(x, 0), (x, h)], fill=(0, 255, 255), width=1)
        if label:
            dr.text((x + 2, 2), str(x), fill=(255, 255, 0))
    for y in range(0, h, step):
        dr.line([(0, y), (w, y)], fill=(0, 255, 255), width=1)
        if label:
            dr.text((2, y + 2), str(y), fill=(255, 255, 0))
    return im


def cmd_render():
    import trimesh
    from PIL import Image
    glb, out = sys.argv[2], sys.argv[3]
    yaws = [float(v) for v in sys.argv[sys.argv.index("--yaws") + 1:]] if "--yaws" in sys.argv else [0.0, 90.0]
    os.makedirs(out, exist_ok=True)
    sc = trimesh.load(glb, process=False)
    g = list(sc.geometry.values())[0] if hasattr(sc, "geometry") else sc
    tex = np.asarray(g.visual.material.baseColorTexture.convert("RGB"))
    H, W = tex.shape[:2]
    uv = np.asarray(g.visual.uv, np.float64)
    ys, xs, pos, nrm = pj.texel_geometry(g, uv[np.asarray(g.faces)], H, W)
    col = tex[ys, xs]
    sub = slice(None, None, 2)
    pos, nrm, col = pos[sub].astype(np.float32), nrm[sub].astype(np.float32), col[sub]
    centre = ((pos.min(0) + pos.max(0)) / 2).astype(np.float64)
    meta = {"glb": glb, "centre": centre.tolist(), "cams": {}}
    for yaw in yaws:
        img, dep, cam = reg.render_splat(pos, col, nrm, centre, yaw)
        name = f"render_yaw{int(yaw)}"
        Image.fromarray(img).save(f"{out}/{name}.png")
        draw_grid(img).save(f"{out}/{name}_grid.png")
        np.save(f"{out}/{name}_depth.npy", dep)
        meta["cams"][f"yaw{int(yaw)}"] = {"yaw": yaw, **cam}
        print("rendered", name, cam)
    json.dump(meta, open(f"{out}/meta.json", "w"), indent=1)


def cmd_grid():
    from PIL import Image
    im = Image.open(sys.argv[2]).convert("RGB")
    draw_grid(im, int(_arg("--step", "100"))).save(sys.argv[3])
    print("grid written", im.size)


def cmd_crop():
    from PIL import Image, ImageDraw
    Image.MAX_IMAGE_PIXELS = None
    src, out = sys.argv[2], sys.argv[3]
    x0, y0, x1, y1 = (int(v) for v in sys.argv[4:8])
    zoom = float(sys.argv[8])
    step = int(sys.argv[9]) if len(sys.argv) > 9 else 25
    im = Image.open(src).convert("RGBA")
    bg = Image.new("RGBA", im.size, (70, 90, 70, 255))
    bg.alpha_composite(im)
    c = bg.convert("RGB").crop((x0, y0, x1, y1)).resize((int((x1 - x0) * zoom), int((y1 - y0) * zoom)), Image.LANCZOS)
    dr = ImageDraw.Draw(c)
    for x in range(-(-x0 // step) * step, x1, step):
        X = (x - x0) * zoom
        major = (x % (step * 2) == 0)
        dr.line([(X, 0), (X, c.height)], fill=(0, 255, 255) if major else (0, 100, 100), width=1)
        if major:
            dr.text((X + 2, 2), str(x), fill=(255, 255, 0))
    for y in range(-(-y0 // step) * step, y1, step):
        Y = (y - y0) * zoom
        major = (y % (step * 2) == 0)
        dr.line([(0, Y), (c.width, Y)], fill=(0, 255, 255) if major else (0, 100, 100), width=1)
        if major:
            dr.text((2, Y + 2), str(y), fill=(255, 255, 0))
    c.save(out)
    print("crop", c.size, "of", im.size)


def cmd_solve():
    from PIL import Image, ImageDraw
    lm = json.load(open(sys.argv[2], encoding="utf-8"))
    out = sys.argv[3]
    os.makedirs(out, exist_ok=True)
    meta = json.load(open(lm["meta"], encoding="utf-8"))
    centre = np.array(meta["centre"])
    mesh3d = {}
    for key, picks in lm["mesh"].items():                    # lift each pick through its render's depth buffer
        cam = meta["cams"][key]
        dep = np.load(os.path.join(os.path.dirname(lm["meta"]), f"render_{key}_depth.npy"))
        for label, (px, py) in picks.items():
            mesh3d[label] = reg.lift(px, py, dep, cam, centre, cam["yaw"]).tolist()
    print("mesh landmarks (3D):", {k: [round(v, 3) for v in p] for k, p in mesh3d.items()})
    cams = {}
    for name, v in lm["views"].items():
        picks = v.get("picks")
        if "mirror_of" in v:
            src = lm["views"][v["mirror_of"]]
            w = Image.open(src["path"]).size[0]
            picks = reg.mirror_picks(src["picks"], w)
        labels = [k for k in picks if k in mesh3d]
        if len(labels) < 4:
            print(f"{name}: only {len(labels)} landmarks, skipped (keeps the silhouette fit)")
            continue
        sol = reg.solve_camera([mesh3d[k] for k in labels], [picks[k] for k in labels], centre, v["yaw"],
                               yaw_range=float(v.get("yaw_range", 15.0)), affine=bool(v.get("affine", lm.get("affine", True))),
                               fix_yaw=bool(v.get("fix_yaw", False)))
        sol["labels"] = labels
        sol["centre"] = centre.tolist()
        cams[name] = sol
        print(f"{name}: {sol['kind']} yaw {v['yaw']:g} -> {sol['yaw']:.1f}, rms {sol['rms_px']:.1f}px, "
              f"per-point {dict(zip(labels, sol['resid_px']))}")
        im = Image.open(v["path"]).convert("RGB")
        dr = ImageDraw.Draw(im)
        rad = max(6, im.size[0] // 120)
        for k, (x, y) in picks.items():                      # green = picked, red = mesh landmark projected by the solved camera
            dr.ellipse([x - rad, y - rad, x + rad, y + rad], outline=(0, 255, 0), width=3)
        px, py, _ = reg.project_cam([mesh3d[k] for k in labels], sol, centre)
        for k, x, y in zip(labels, px, py):
            dr.line([x - rad, y, x + rad, y], fill=(255, 0, 0), width=3)
            dr.line([x, y - rad, x, y + rad], fill=(255, 0, 0), width=3)
            dr.text((x + rad + 2, y - rad), k, fill=(255, 255, 0))
        im.save(os.path.join(out, "overlay_" + os.path.splitext(os.path.basename(name))[0] + ".jpg"), quality=88)
    json.dump({"mesh3d": mesh3d, "cams": cams}, open(os.path.join(out, "cameras.json"), "w"), indent=1)
    print("cameras.json written")


if __name__ == "__main__":
    {"render": cmd_render, "grid": cmd_grid, "crop": cmd_crop, "solve": cmd_solve}[sys.argv[1]]()
