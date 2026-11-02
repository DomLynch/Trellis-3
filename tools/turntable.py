"""12-angle turntable contact sheet for a GLB, CPU only (Blender Cycles, no GPU).

    python tools/turntable.py mesh.glb [out_dir] [--size 512] [--samples 24] [--blender PATH]

Writes <out_dir>/frame_00..11.png and <out_dir>/contact.jpg (4 x 3, each cell labelled with its yaw).
Yaw 0 is the model's front (the camera looks along +Y at the -Y face after the glTF Y-up -> Blender Z-up import),
so cells run front, 30, 60, profile (90), ... back (180) ... Also prints the near-black texel share of the
base colour texture (texels covered by a face in UV space, luminance < 8), the number the Factory tracks.
"""
import json
import os
import subprocess
import sys

BLENDER = r"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe"
N = 12


def _render(argv):
    import bpy
    from mathutils import Vector
    glb, out, size, samples = argv[0], argv[1], int(argv[2]), int(argv[3])
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=glb)
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    lo = Vector((1e9,) * 3)
    hi = Vector((-1e9,) * 3)
    for o in meshes:
        for c in o.bound_box:
            w = o.matrix_world @ Vector(c)
            lo = Vector(map(min, lo, w))
            hi = Vector(map(max, hi, w))
    centre, ext = (lo + hi) / 2, max(hi - lo)
    pivot = bpy.data.objects.new("pivot", None)
    bpy.context.scene.collection.objects.link(pivot)
    pivot.location = centre
    for o in meshes:
        if o.parent is None:
            o.parent = pivot
            o.matrix_parent_inverse = pivot.matrix_world.inverted()
    scn = bpy.context.scene
    scn.render.engine = "CYCLES"
    scn.cycles.device = "CPU"
    scn.cycles.samples = samples
    scn.cycles.use_denoising = False
    scn.render.resolution_x = scn.render.resolution_y = size
    scn.render.film_transparent = False
    w = bpy.data.worlds.new("w")
    w.use_nodes = True
    w.node_tree.nodes["Background"].inputs[0].default_value = (0.32, 0.32, 0.34, 1)
    w.node_tree.nodes["Background"].inputs[1].default_value = 1.0
    scn.world = w
    cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
    scn.collection.objects.link(cam)
    cam.data.lens = 60
    d = ext * 2.6
    cam.location = centre + Vector((0, -d, ext * 0.08))
    tgt = bpy.data.objects.new("tgt", None)
    scn.collection.objects.link(tgt)
    tgt.location = centre
    con = cam.constraints.new("TRACK_TO")
    con.target, con.track_axis, con.up_axis = tgt, "TRACK_NEGATIVE_Z", "UP_Y"
    scn.camera = cam
    for name, loc, energy in (("key", (-1.5, -2, 2), 900), ("fill", (2.5, -1.5, 0.5), 350), ("rim", (0, 3, 2), 500)):
        L = bpy.data.objects.new(name, bpy.data.lights.new(name, "AREA"))
        L.data.energy = energy * (ext ** 2)
        L.data.size = ext * 2
        L.location = centre + Vector(loc) * ext
        c2 = L.constraints.new("TRACK_TO")
        c2.target, c2.track_axis, c2.up_axis = tgt, "TRACK_NEGATIVE_Z", "UP_Y"
        scn.collection.objects.link(L)
    import math
    for i in range(N):
        pivot.rotation_euler = (0, 0, math.radians(-360 * i / N))
        scn.render.filepath = os.path.join(out, f"frame_{i:02d}.png")
        bpy.ops.render.render(write_still=True)


def near_black(glb, lum=8):
    import cv2
    import numpy as np
    import trimesh
    sc = trimesh.load(glb, process=False)
    g = list(sc.geometry.values())[0] if hasattr(sc, "geometry") else sc
    tex = np.asarray(g.visual.material.baseColorTexture.convert("RGB"))
    H, W = tex.shape[:2]
    uv, faces = np.asarray(g.visual.uv, np.float64), np.asarray(g.faces)
    m = np.zeros((H, W), np.uint8)
    for tri in faces:
        p = np.stack([uv[tri][:, 0] * (W - 1), (1 - uv[tri][:, 1]) * (H - 1)], 1)
        cv2.fillConvexPoly(m, p.round().astype(np.int32), 255)
    return round(100.0 * ((tex.mean(2) < lum) & (m > 0)).sum() / max(1, (m > 0).sum()), 3)


def sheet(out):
    from PIL import Image, ImageDraw
    fr = [Image.open(os.path.join(out, f"frame_{i:02d}.png")).convert("RGB") for i in range(N)]
    w, h = fr[0].size
    sh = Image.new("RGB", (4 * w, 3 * h))
    dr = ImageDraw.Draw(sh)
    for i, im in enumerate(fr):
        x, y = (i % 4) * w, (i // 4) * h
        sh.paste(im, (x, y))
        dr.text((x + 8, y + 8), f"{i * 360 // N} deg", fill=(255, 255, 0))
    p = os.path.join(out, "contact.jpg")
    sh.save(p, quality=90)
    return p


def main():
    if "--bpy" in sys.argv:
        _render(sys.argv[len(sys.argv) - sys.argv[::-1].index("--"):])
        return
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    opt = lambda k, d: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d  # noqa: E731
    glb = a[0]
    flagvals = {opt(k, None) for k in ("--size", "--samples", "--blender")} - {None}
    a = [x for x in a if x not in flagvals]
    glb = a[0]
    out = a[1] if len(a) > 1 else os.path.join(os.path.dirname(os.path.abspath(glb)), "turntable12")
    size, samples, blender = opt("--size", "512"), opt("--samples", "24"), opt("--blender", BLENDER)
    out = os.path.abspath(out)
    os.makedirs(out, exist_ok=True)
    subprocess.run([blender, "--background", "--python", os.path.abspath(__file__), "--", "--bpy", "--",
                    os.path.abspath(glb), out, size, samples], check=True,
                   stdout=subprocess.DEVNULL)
    p = sheet(out)
    try:
        nb = near_black(glb)
    except Exception as e:  # noqa: BLE001
        nb = f"n/a ({e})"
    json.dump({"glb": glb, "contact": p, "near_black_pct": nb}, open(os.path.join(out, "turntable.json"), "w"), indent=1)
    print(json.dumps({"contact": p, "near_black_pct": nb}))


if __name__ == "__main__":
    main()
