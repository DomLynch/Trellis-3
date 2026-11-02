"""
Pixal3D (TencentARC, SIGGRAPH 2026, MIT) as a Trellis-3 preset.

    python generate.py front.png --preset pixal3d --output out/head                      # single view
    python generate.py front.png tq.png prof.png back.png --preset pixal3d \
        --yaws 0 45 90 180 --output out/head                                              # posed multi-view

Pixal3D has its own venv, repo and weights, so this module only launches tools/pixal3d_entry.py inside that
venv and then lays the result out like every other Trellis-3 run: <output>.glb, <output>_basecolor.png,
<output>_normal.png / _metallic_roughness.png when present, and <output>.receipt.json.

Multi-view mode (inference_mv.py + the *_mv weights) wants camera poses. A view directory with transforms.json
is written from the yaw of each image (camera convention of Pixal3D's example: azimuth 0 = front camera at -Y,
azimuth 90 = camera on +X, Z up). Each image is cut out, centred on its silhouette and padded to a square at a
common height ratio, because the pose file assumes the object fills the frame like the training renders.

Paths default to this machine's layout and can be overridden with environment variables:
PIXAL3D_PY, PIXAL3D_REPO, PIXAL3D_RUN, PIXAL3D_DINO, PIXAL3D_HF_HOME, PIXAL3D_TORCH_HOME, PIXAL3D_SHIMS.
"""
import json
import math
import os
import shutil
import subprocess
import sys
import time

DOCS = os.path.join(os.path.expanduser("~"), "Documents")
DEFAULTS = {
    "PIXAL3D_PY": os.path.join(DOCS, "Pixal3D-312", ".venv", "Scripts", "python.exe"),
    "PIXAL3D_REPO": os.path.join(DOCS, "Pixal3D"),
    "PIXAL3D_RUN": os.path.join(DOCS, "Pixal3D-run"),           # patched pipeline.json / pipeline_mv.json + ckpts junction
    "PIXAL3D_DINO": os.path.join(DOCS, "models", "dinov3-vitl16-pretrain-lvd1689m"),
    "PIXAL3D_HF_HOME": os.path.join(DOCS, "models", "hf-cache"),
    "PIXAL3D_TORCH_HOME": os.path.join(DOCS, "models", "torch-hub"),
    "PIXAL3D_SHIMS": r"C:\HANDOVER-Macbook\extracted\7-repos\cf-natten\tools\pixal3d_shims",
}
ENTRY = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools", "pixal3d_entry.py")

# Pixal3D's own mv example: 20 degree horizontal fov, camera distance 3.1192
FOV_X = 0.3490658503988659
DISTANCE = 3.1192049980163574
FILL = 0.88          # subject height as a share of the square view


def conf(key):
    return os.environ.get(key, DEFAULTS[key])


def camera_matrix(yaw_deg, distance=DISTANCE):
    """Blender/NeRF camera-to-world for azimuth `yaw_deg` (Z-up world, camera looks -Z, +Y up)."""
    a = math.radians(yaw_deg)
    c, s = math.cos(a), math.sin(a)
    return [[c, 0.0, s, distance * s],
            [s, 0.0, -c, -distance * c],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]]


def default_yaws(n):
    """front, three-quarter, profile, back, then the mirror three-quarter and profile."""
    return [0, 45, 90, 180, 315, 270][:n]


def frame_view(rgba, side=1024, fill=FILL):
    """Centre the subject's alpha bbox in a square canvas so its height is `fill` of the side."""
    from PIL import Image
    bbox = rgba.getchannel("A").point(lambda v: 255 if v > 127 else 0).getbbox()
    if bbox is None:
        raise ValueError("view has an empty cut-out")
    crop = rgba.crop(bbox)
    k = side * fill / crop.height
    w, h = max(1, round(crop.width * k)), max(1, round(crop.height * k))
    if w > side:
        k, w, h = side / crop.width, side, max(1, round(crop.height * side / crop.width))
    crop = crop.resize((w, h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    canvas.paste(crop, ((side - w) // 2, (side - h) // 2), crop)
    return canvas


def write_views_dir(images, yaws, out_dir, cutout=None):
    """images: paths (RGBA preferred). Returns out_dir. Frame 0 must be the front view (yaw 0)."""
    from PIL import Image
    if len(images) != len(yaws):
        raise ValueError(f"{len(images)} images but {len(yaws)} yaws")
    if abs(yaws[0]) > 1e-6:
        raise ValueError("the first view is the main view and must have yaw 0 (front)")
    os.makedirs(out_dir, exist_ok=True)
    frames = []
    for i, (p, yaw) in enumerate(zip(images, yaws)):
        im = Image.open(p).convert("RGBA")
        if im.getchannel("A").getextrema()[0] == 255:
            raise ValueError(f"{p} has no cut-out alpha; run the cut-out step (tools/cutout_cpu.py) first")
        name = f"view{i:02d}_azim{int(round(yaw)) % 360:03d}.png"
        frame_view(im).save(os.path.join(out_dir, name))
        frames.append({"file_path": name, "name": f"azim{int(round(yaw)) % 360:03d}",
                       "transform_matrix": camera_matrix(yaw)})
    json.dump({"camera_angle_x": FOV_X, "mesh_scale": 1.0, "frames": frames},
              open(os.path.join(out_dir, "transforms.json"), "w"), indent=2)
    return out_dir


def build_command(images, output, yaws=None, seed=42, resolution=-1, low_vram=True, views_dir=None, glb="cpu"):
    """Command line for tools/pixal3d_entry.py under the Pixal3D venv."""
    cmd = [conf("PIXAL3D_PY"), "-u", ENTRY, "--repo", conf("PIXAL3D_REPO"), "--run-dir", conf("PIXAL3D_RUN"),
           "--dino-dir", conf("PIXAL3D_DINO"), "--output", output + ".glb", "--seed", str(seed),
           "--resolution", str(resolution), "--glb", glb]
    if low_vram:
        cmd.append("--low-vram")
    if len(images) > 1:
        cmd += ["--views-dir", views_dir, "--num-views", str(len(images))]
    else:
        cmd += ["--image", images[0]]
    return cmd


def build_env():
    env = dict(os.environ)
    env.update({"HF_HOME": conf("PIXAL3D_HF_HOME"), "HF_HUB_OFFLINE": "1", "ATTN_BACKEND": "sdpa",
                "SPARSE_ATTN_BACKEND": "sdpa", "TORCH_HOME": conf("PIXAL3D_TORCH_HOME"),
                "PYTHONPATH": conf("PIXAL3D_SHIMS") + os.pathsep + env.get("PYTHONPATH", ""),
                "PYTHONUNBUFFERED": "1"})
    return env


def extract_textures(glb_path, prefix):
    """Write the GLB's PBR images next to it: <prefix>_basecolor.png, _normal.png, _metallic_roughness.png."""
    import trimesh
    sc = trimesh.load(glb_path)
    g = list(sc.geometry.values())[0] if hasattr(sc, "geometry") else sc
    m = g.visual.material
    out = []
    for attr, suffix in (("baseColorTexture", "basecolor"), ("normalTexture", "normal"),
                         ("metallicRoughnessTexture", "metallic_roughness"), ("emissiveTexture", "emissive")):
        im = getattr(m, attr, None)
        if im is not None:
            p = f"{prefix}_{suffix}.png"
            im.save(p)
            out.append(p)
    return out


def run_from_args(args, log=print):
    """generate.py --preset pixal3d. Returns the process exit code."""
    from trellis3.receipt import write_receipt
    images = list(args.images)
    yaws = list(args.yaws) if getattr(args, "yaws", None) else default_yaws(len(images))
    if len(yaws) != len(images):
        log(f"Error: {len(images)} images but {len(yaws)} --yaws")
        return 1
    out = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    for key in ("PIXAL3D_PY", "PIXAL3D_REPO", "PIXAL3D_RUN"):
        if not os.path.exists(conf(key)):
            log(f"Error: {key}={conf(key)} not found (see docs/multi-image.md)")
            return 1
    views_dir = None
    if len(images) > 1:
        views_dir = write_views_dir(images, yaws, out + "_views")
        log(f"Pixal3D multi-view: {len(images)} views, yaws {yaws}, views dir {views_dir}")
    cmd = build_command(images, out, yaws, seed=args.seed, views_dir=views_dir,
                        resolution=1024 if args.pipeline_type in (None, "1024_cascade") else -1)
    t0 = time.time()
    log("Launching Pixal3D: " + " ".join(cmd[2:]))
    code = subprocess.run(cmd, env=build_env(), cwd=conf("PIXAL3D_REPO")).returncode
    wall = time.time() - t0
    glb = out + ".glb"
    if code != 0 or not os.path.exists(glb):
        log(f"Pixal3D failed (exit {code})")
        return code or 1
    files = [glb] + extract_textures(glb, out)
    if not args.no_receipt:
        rp = write_receipt(out, {"preset": "pixal3d", "seed": args.seed, "images": images, "yaws": yaws,
                                 "mode": "multiview" if len(images) > 1 else "single",
                                 "licence": "Pixal3D MIT (Tencent); DINOv3 Meta licence; BiRefNet MIT; NAF Apache-2.0"},
                           images[0], files, {"pixal3d": round(wall, 1)})
        log(f"Receipt: {rp}")
    log(f"\nPixal3D done in {wall:.0f}s -> {glb}")
    return 0
