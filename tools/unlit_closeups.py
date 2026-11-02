"""Unlit face close-ups of a GLB (Blender Cycles on the CPU, emission shader, Standard view transform, no lights).

    python tools/unlit_closeups.py <mesh.glb> <out_prefix> <cameras.json | auto:fx,fy,fz,span> [--size 2048] [--yaws 0 45] [--samples 8] [--mode unlit|grey]

--mode grey renders the shape only: mid-grey material (albedo 0.30) under three area lights. auto:fx,fy,fz,span frames from the mesh bbox
(centre as bbox fractions, span as a share of the bbox height) when there are no landmarks, e.g. for another model's mesh.

Writes <out_prefix>_front.png, <out_prefix>_yaw45.png, ... : orthographic, framed on the face (eyes, nose, mouth, chin) from the 3D
landmarks in cameras.json (tools/face_register.py solve). The base-colour texture is shown exactly as stored: no lights, no normal
map, so lighting can neither hide nor add features (ghosting, double moustaches, misplaced eyes stay visible).
"""
import json
import math
import os
import subprocess
import sys

BLENDER = r"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe"


def _bpy(argv):
    import bpy
    from mathutils import Vector
    glb, prefix, cams, size, samples = argv[0], argv[1], argv[2], int(argv[3]), int(argv[4])
    yaws = [float(v) for v in argv[5:]]
    mode = os.environ.get("CLOSEUP_MODE", "unlit")            # unlit = base colour emission; grey = lit clay (albedo 0.30, shape only)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=glb)
    if cams.startswith("auto:"):
        fx, fy, fz, fs = (float(v) for v in cams[5:].split(","))
        lo = Vector((1e9,) * 3)
        hi = Vector((-1e9,) * 3)
        for ob in bpy.context.scene.objects:
            if ob.type == "MESH":
                for c in ob.bound_box:
                    w = ob.matrix_world @ Vector(c)
                    lo = Vector(map(min, lo, w))
                    hi = Vector(map(max, hi, w))
        # Blender (x, y, z) = glTF (x, -z, y): convert the bbox back to glTF axes
        g_lo, g_hi = (lo.x, lo.z, -hi.y), (hi.x, hi.z, -lo.y)
        cx, cy, cz = (g_lo[0] + fx * (g_hi[0] - g_lo[0]), g_lo[1] + fy * (g_hi[1] - g_lo[1]), g_lo[2] + fz * (g_hi[2] - g_lo[2]))
        span = fs * (g_hi[1] - g_lo[1]) / 1.9
    else:
        lm = json.load(open(cams, encoding="utf-8"))["mesh3d"]
        pts = [lm[k] for k in ("eye_vl", "eye_vr", "mouth_c", "chin", "glabella") if k in lm]
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        cz = sum(p[2] for p in pts) / len(pts)
        ed = abs(lm["eye_vr"][0] - lm["eye_vl"][0])
        span = max(max(p[1] for p in pts) - min(p[1] for p in pts), 2 * ed)
    for ob in list(bpy.context.scene.objects):
        if ob.type != "MESH":
            continue
        for slot in ob.material_slots:
            m = slot.material
            img = None
            if m and m.node_tree:
                for n in m.node_tree.nodes:
                    if n.type == "TEX_IMAGE" and n.image and "basecolor" in (n.label + n.name + n.image.name).lower() or \
                            (n.type == "TEX_IMAGE" and n.image and img is None):
                        img = n.image
                bsdf = next((n for n in m.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
                if bsdf and bsdf.inputs["Base Color"].links:
                    src = bsdf.inputs["Base Color"].links[0].from_node
                    if src.type == "TEX_IMAGE" and src.image:
                        img = src.image
            nm = bpy.data.materials.new("unlit")
            nm.use_nodes = True
            nt = nm.node_tree
            nt.nodes.clear()
            out = nt.nodes.new("ShaderNodeOutputMaterial")
            if mode == "grey":
                bs = nt.nodes.new("ShaderNodeBsdfPrincipled")
                bs.inputs["Base Color"].default_value = (0.30, 0.30, 0.30, 1.0)
                bs.inputs["Roughness"].default_value = 0.6
                bs.inputs["Metallic"].default_value = 0.0
                nt.links.new(bs.outputs["BSDF"], out.inputs["Surface"])
                slot.material = nm
                continue
            em = nt.nodes.new("ShaderNodeEmission")
            if img is not None:
                tx = nt.nodes.new("ShaderNodeTexImage")
                tx.image = img
                tx.interpolation = "Cubic"
                uvn = nt.nodes.new("ShaderNodeUVMap")
                nt.links.new(uvn.outputs["UV"], tx.inputs["Vector"])
                nt.links.new(tx.outputs["Color"], em.inputs["Color"])
            nt.links.new(em.outputs["Emission"], out.inputs["Surface"])
            slot.material = nm
    scn = bpy.context.scene
    if mode == "grey":
        tgt0 = Vector((cx, -cz, cy))
        for name, loc, energy in (("key", (-1.2, -1.6, 1.0), 260), ("fill", (1.8, -1.0, 0.2), 90), ("rim", (0.0, 1.6, 1.0), 120)):
            L = bpy.data.objects.new(name, bpy.data.lights.new(name, "AREA"))
            L.data.energy = energy
            L.data.size = 1.5
            L.location = tgt0 + Vector(loc)
            L.rotation_euler = (tgt0 - L.location).to_track_quat("-Z", "Y").to_euler()
            scn.collection.objects.link(L)
    scn.render.engine = "CYCLES"
    scn.cycles.device = "CPU"
    scn.cycles.samples = samples
    scn.cycles.use_denoising = False
    scn.view_settings.view_transform = "Standard"
    scn.render.resolution_x = scn.render.resolution_y = size
    w = bpy.data.worlds.new("w")
    w.use_nodes = True
    w.node_tree.nodes["Background"].inputs[0].default_value = (0.2, 0.2, 0.2, 1)
    scn.world = w
    cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
    cam.data.type = "ORTHO"
    cam.data.ortho_scale = span * 1.9
    cam.data.clip_start, cam.data.clip_end = 0.01, 20
    scn.collection.objects.link(cam)
    scn.camera = cam
    tgt = Vector((cx, -cz, cy))                                  # glTF (x, y, z) -> Blender (x, -z, y)
    for yaw in yaws:
        a = math.radians(yaw)
        d = Vector((math.sin(a), -math.cos(a), 0.0))             # direction to the camera, glTF (sin, 0, cos)
        cam.location = tgt + d * 3.0
        direction = tgt - cam.location
        cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
        scn.render.filepath = f"{prefix}_{mode}_{'front' if yaw == 0 else 'yaw%d' % int(yaw)}.png"
        bpy.ops.render.render(write_still=True)


def main():
    if "--bpy" in sys.argv:
        _bpy(sys.argv[sys.argv.index("--bpy") + 1:])
        return
    glb, prefix, cams = sys.argv[1:4]
    mode = sys.argv[sys.argv.index("--mode") + 1] if "--mode" in sys.argv else "unlit"
    size = sys.argv[sys.argv.index("--size") + 1] if "--size" in sys.argv else "2048"
    samples = sys.argv[sys.argv.index("--samples") + 1] if "--samples" in sys.argv else "8"
    yaws = sys.argv[sys.argv.index("--yaws") + 1:] if "--yaws" in sys.argv else ["0", "45"]
    yaws = [y for y in yaws if not y.startswith("--")]
    if "--mode" in yaws:
        yaws = yaws[:yaws.index("--mode")]
    os.makedirs(os.path.dirname(os.path.abspath(prefix)), exist_ok=True)
    subprocess.run([BLENDER, "--background", "--python", os.path.abspath(__file__), "--", "--bpy", os.path.abspath(glb),
                    os.path.abspath(prefix), cams if cams.startswith("auto:") else os.path.abspath(cams), size, samples] + yaws,
                   check=True, stdout=subprocess.DEVNULL, env=dict(os.environ, CLOSEUP_MODE=mode))
    print("closeups written with prefix", prefix)


if __name__ == "__main__":
    main()
