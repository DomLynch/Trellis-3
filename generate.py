"""
Generate a 3D mesh from a single image using TRELLIS.2 on Apple Silicon.

Trellis-3 additions over upstream trellis-mac:
  --preset draft|final   fast throwaway runs vs. full quality
  --max-tokens N         cap cascade tokens (the main 1024/1536 memory lever)
  --resident             keep models on the GPU instead of shuttling via CPU
  --no-thermal-guard     skip the pmset throttle check before generating
  --pipeline-type 1536_cascade and --texture-size 4096 are accepted; both
                         need far more memory than a 24GB machine has
Only the checkpoints the chosen pipeline type needs are loaded.
"""

import os
import sys

from trellis3.runner import bootstrap
bootstrap()  # must precede any torch import

import argparse
import time

from PIL import Image as PILImage

from trellis3 import memory
from trellis3.export import export_mesh
from trellis3.presets import PRESETS, apply_preset
from trellis3.receipt import write_receipt
from trellis3.runner import Runner, WatchdogError
from trellis3.thermal import wait_until_cool


def build_parser():
    p = argparse.ArgumentParser(description="Generate 3D mesh from an image using TRELLIS.2")
    p.add_argument("image", help="Path to input image")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    p.add_argument("--output", default="output_3d", help="Output filename without extension")
    p.add_argument("--preset", choices=sorted(PRESETS), default=None,
                   help="draft: 512/8 steps/no texture. final: 1024_cascade/12 steps/2048 texture")
    p.add_argument("--pipeline-type", default=None, choices=list(memory.PIPELINE_TYPES),
                   help="Pipeline resolution (default: 512, or the preset's)")
    p.add_argument("--texture-size", type=int, default=None, choices=[512, 1024, 2048, 4096],
                   help="PBR texture resolution (default: 1024, or the preset's). 4096 is untested on 24GB")
    p.add_argument("--no-texture", action="store_true", help="Skip texture baking, export geometry only")
    p.add_argument("--steps", type=int, default=None,
                   help="Sampler steps for all three flow phases (default: pipeline JSON, 12)")
    p.add_argument("--max-tokens", type=int, default=49152,
                   help="Max cascade tokens; lower it to cut 1024/1536 memory (default: 49152)")
    p.add_argument("--resident", action="store_true",
                   help="Keep loaded models on the GPU (low_vram=False). Opt-in, unmeasured on 24GB")
    p.add_argument("--no-stage-hooks", action="store_true",
                   help="Do not flush the MPS allocator cache between stages")
    p.add_argument("--no-thermal-guard", action="store_true",
                   help="Do not wait for CPU_Speed_Limit to recover before generating")
    bake = p.add_argument_group("bake (Metal path)")
    bake.add_argument("--metallic-factor", type=float, default=0.0,
                      help="Pin glTF metallicFactor (upstream 1.0; default 0.0 = non-metal). -1 leaves upstream")
    bake.add_argument("--gamma", type=float, default=None,
                      help="Apply 1/gamma to base colour, e.g. 2.2 if output looks too dark")
    bake.add_argument("--no-fix-winding", action="store_true",
                      help="Skip the consistent-winding pass before baking")
    bake.add_argument("--uv-cone-deg", type=float, default=None,
                      help="UV chart cone half-angle (upstream 90); larger = fewer charts")
    bake.add_argument("--uv-smooth", type=float, default=None,
                      help="UV chart smoothing strength (upstream 1); larger = fewer charts")
    p.add_argument("--no-receipt", action="store_true", help="Do not write <output>.receipt.json")
    return p


def bake_kwargs(args):
    return {
        "metallic_factor": None if args.metallic_factor < 0 else args.metallic_factor,
        "gamma": args.gamma,
        "fix_winding": not args.no_fix_winding,
        "uv_cone_deg": args.uv_cone_deg,
        "uv_smooth": args.uv_smooth,
    }


def main(argv=None):
    args = build_parser().parse_args(argv)

    if not os.path.exists(args.image):
        print(f"Error: {args.image} not found")
        sys.exit(1)

    cfg = apply_preset(args.preset, pipeline_type=args.pipeline_type, steps=args.steps,
                       texture_size=args.texture_size, no_texture=args.no_texture)

    print("=" * 60)
    print("TRELLIS.2 on Apple Silicon (Trellis-3)")
    print("=" * 60)
    print(f"preset={args.preset or '-'} pipeline={cfg['pipeline_type']} steps={cfg['steps'] or 'default'} "
          f"texture={'none' if cfg['no_texture'] else cfg['texture_size']} max_tokens={args.max_tokens}")

    runner = Runner(cfg["pipeline_type"], resident=args.resident,
                    stage_hooks=not args.no_stage_hooks)

    img = PILImage.open(args.image)
    print(f"Input: {args.image} ({img.size[0]}x{img.size[1]})")

    if not args.no_thermal_guard:
        wait_until_cool()

    print(f"\nGenerating 3D model (pipeline={cfg['pipeline_type']}, seed={args.seed})...")
    t0 = time.time()
    try:
        mesh = runner.generate(img, seed=args.seed, pipeline_type=cfg["pipeline_type"],
                               steps=cfg["steps"], max_num_tokens=args.max_tokens)
    except WatchdogError as e:
        print(str(e))
        sys.exit(2)
    t_gen = time.time() - t0
    print(f"\nMesh: {mesh.vertices.shape[0]:,} vertices, {mesh.faces.shape[0]:,} triangles")
    print(f"Generation time: {t_gen:.1f}s")

    t1 = time.time()
    outputs = export_mesh(mesh, args.output, texture_size=cfg["texture_size"],
                          no_texture=cfg["no_texture"], **bake_kwargs(args))
    t_bake = time.time() - t1

    rss = memory.rss_gb()
    if rss:
        print(f"Peak RSS: {rss:.1f} GB")
    if not args.no_receipt:
        settings = {**cfg, "seed": args.seed, "max_num_tokens": args.max_tokens,
                    "resident": args.resident, **bake_kwargs(args)}
        rp = write_receipt(args.output, settings, args.image, outputs,
                           {"generate": round(t_gen, 1), "bake": round(t_bake, 1)}, peak_rss_gb=rss)
        print(f"Receipt: {rp}")
    print(f"\nTotal time: {time.time() - t0:.1f}s generation + baking")


if __name__ == "__main__":
    main()
