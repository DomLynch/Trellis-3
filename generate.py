"""
Generate a 3D mesh from a single image using TRELLIS.2 on Apple Silicon.

Trellis-3 additions over upstream trellis-mac:
  --preset draft|final|master   throwaway / full quality / desktop master
  staged sampling               latents saved after every stage to <output>.latents.pt;
                                --resume continues, --from-latents decodes in a fresh
                                process that loads only the two decoders
  --two-stage                   sample here, decode in a child process (sampler memory gone)
  --ss/shape/tex-steps          per-stage step counts
  --min-free-gb N               stop cleanly between steps when free RAM drops under N
  --max-tokens N                cap cascade tokens (the main 1024/1536 memory lever)
  --attn-chunk N                query-chunked full attention
  --profile-memory              per-stage memory table in the receipt
Only the checkpoints the chosen pipeline type needs are loaded, and each
model is dropped once its stage is done.
"""

import os
import sys

from trellis3.runner import bootstrap
bootstrap()  # must precede any torch import

import argparse
import subprocess
import time

from PIL import Image as PILImage

from trellis3 import memory
from trellis3.export import export_mesh
from trellis3.latents import describe, load_state, save_state
from trellis3.presets import PRESETS, apply_preset
from trellis3.receipt import write_receipt
from trellis3.runner import Runner, WatchdogError
from trellis3.stages import MemoryPressure
from trellis3.thermal import wait_until_cool


def _gamma_arg(v):
    if v.lower() == "auto":
        return "auto"
    return float(v)


def build_parser():
    p = argparse.ArgumentParser(description="Generate 3D mesh from an image using TRELLIS.2")
    p.add_argument("image", help="Path to input image")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    p.add_argument("--output", default="output_3d", help="Output filename without extension")
    p.add_argument("--preset", choices=sorted(PRESETS), default=None,
                   help="draft: 512/8 steps/no texture. final: 1024_cascade/2048. "
                        "master: 1024_cascade/4096/1M faces/normal map/two-stage. "
                        "hero: 1536_cascade/8192/2M faces (64 GB machines)")
    p.add_argument("--pipeline-type", default=None, choices=list(memory.PIPELINE_TYPES),
                   help="Pipeline resolution (default: 512, or the preset's)")
    p.add_argument("--texture-size", type=int, default=None, choices=[512, 1024, 2048, 4096, 8192],
                   help="PBR texture resolution (default: 1024, or the preset's)")
    p.add_argument("--no-texture", action="store_true", help="Skip texture baking, export geometry only")
    p.add_argument("--bake-faces", type=int, default=None,
                   help="Face count of the baked/exported mesh (default 200000; master preset 1000000)")
    p.add_argument("--steps", type=int, default=None, help="Sampler steps for all three stages (default 12)")
    p.add_argument("--ss-steps", type=int, default=None, help="Steps for the sparse-structure stage")
    p.add_argument("--shape-steps", type=int, default=None, help="Steps for the shape stage")
    p.add_argument("--tex-steps", type=int, default=None, help="Steps for the texture stage (the slow one at 1024)")
    p.add_argument("--max-tokens", type=int, default=49152,
                   help="Max cascade tokens; lower it to cut 1024/1536 memory (default: 49152)")

    run = p.add_argument_group("run control")
    run.add_argument("--two-stage", action="store_true",
                     help="Sample in this process, then decode+export in a fresh child process")
    run.add_argument("--single-process", action="store_true",
                     help="Decode in this process even if the preset asks for two-stage (fine on 64 GB)")
    run.add_argument("--attn-auto-gb", type=float, default=None,
                     help="Tile attention automatically when the score matrix would exceed this many GB (default 6; 0 = never)")
    run.add_argument("--from-latents", metavar="PATH", default=None,
                     help="Skip sampling: decode and export the saved latents (loads decoders only)")
    run.add_argument("--resume", action="store_true",
                     help="Continue from <latents> if it exists (completed stages are skipped)")
    run.add_argument("--latents", metavar="PATH", default=None,
                     help="Where to save latents (default: <output>.latents.pt)")
    run.add_argument("--no-save-latents", action="store_true", help="Do not write latents after each stage")
    run.add_argument("--min-free-gb", type=float, default=0.0,
                     help="Stop cleanly between sampler steps if free RAM drops under this (0 = off)")
    run.add_argument("--keep-models", action="store_true",
                     help="Do not drop models after their stage (default drops them)")
    run.add_argument("--legacy-run", action="store_true",
                     help="Use upstream pipeline.run() in one call instead of staged sampling")
    run.add_argument("--resident", action="store_true",
                     help="Keep loaded models on the GPU (low_vram=False). Opt-in, unmeasured on 24GB")
    run.add_argument("--no-stage-hooks", action="store_true",
                     help="Do not flush the MPS allocator cache between stages")
    run.add_argument("--no-thermal-guard", action="store_true",
                     help="Do not wait for CPU_Speed_Limit to recover before generating")
    run.add_argument("--attn-chunk", type=int, default=0,
                     help="Split full attention into query chunks of N rows (0 = off)")
    run.add_argument("--attn-tile", default=None, metavar="LQ,SK",
                     help="Exact tiled attention with bounded memory, e.g. 2048,4096. Use this for 1024 on a 24GB Mac")
    run.add_argument("--profile-memory", action="store_true",
                     help="Sample RSS/MPS memory per stage; print a table and store it in the receipt")

    bake = p.add_argument_group("bake (Metal path)")
    bake.add_argument("--metallic-factor", type=float, default=0.0,
                      help="Pin glTF metallicFactor (upstream 1.0; default 0.0 = non-metal). -1 leaves upstream")
    bake.add_argument("--gamma", type=_gamma_arg, default="auto",
                      help="Base-colour gamma: 'auto' matches the input photo (default), or a number e.g. 1.7, or 0 for none")
    bake.add_argument("--no-fix-winding", action="store_true",
                      help="Skip the consistent-winding pass before baking")
    bake.add_argument("--uv-cone-deg", type=float, default=None,
                      help="UV chart cone half-angle (upstream 90); larger = fewer charts")
    bake.add_argument("--uv-smooth", type=float, default=None,
                      help="UV chart smoothing strength (upstream 1); larger = fewer charts")
    bake.add_argument("--normal-map", action="store_true",
                      help="Bake a tangent-space normal map from the full-res decoder mesh (CPU)")
    bake.add_argument("--normal-flip-v", action="store_true",
                      help="Flip V when baking the normal map (if it looks vertically wrong)")
    bake.add_argument("--no-snap-hires", action="store_true",
                      help="KDTree path: don't snap texels onto the full-res mesh before sampling")
    p.add_argument("--no-receipt", action="store_true", help="Do not write <output>.receipt.json")
    return p


def bake_kwargs(args, cfg):
    gamma = args.gamma
    if gamma == 0:
        gamma = None
    return {
        "metallic_factor": None if args.metallic_factor < 0 else args.metallic_factor,
        "gamma": gamma,
        "fix_winding": not args.no_fix_winding,
        "uv_cone_deg": args.uv_cone_deg,
        "uv_smooth": args.uv_smooth,
        "normal_map": cfg["normal_map"],
        "normal_flip_v": args.normal_flip_v,
        "snap_hires": not args.no_snap_hires,
        "bake_faces": cfg["bake_faces"],
    }


def steps_spec(args, cfg):
    """Per-stage steps; None means the pipeline default (12)."""
    base = cfg["steps"]
    spec = {"ss": args.ss_steps or base, "shape": args.shape_steps or base, "tex": args.tex_steps or base}
    return None if not any(spec.values()) else spec


def _export_and_finish(args, cfg, mesh, t0, t_gen, profiler, extra_settings):
    if profiler:
        profiler.mark("export")
    t1 = time.time()
    outputs = export_mesh(mesh, args.output, texture_size=cfg["texture_size"],
                          no_texture=cfg["no_texture"], reference_image=args.image,
                          **bake_kwargs(args, cfg))
    t_bake = time.time() - t1

    rss = memory.rss_gb()
    if rss:
        print(f"Peak RSS: {rss:.1f} GB")
    extra = {}
    if profiler:
        from trellis3.profile import activate
        prof = profiler.stop()
        activate(None)
        print("\nMemory profile (peak per stage):\n" + profiler.table())
        extra["memory_profile"] = prof
    if not args.no_receipt:
        settings = {**cfg, "seed": args.seed, "max_num_tokens": args.max_tokens,
                    "resident": args.resident, "attn_chunk": args.attn_chunk, "attn_tile": args.attn_tile,
                    "steps_per_stage": steps_spec(args, cfg), **bake_kwargs(args, cfg), **extra_settings}
        rp = write_receipt(args.output, settings, args.image, outputs,
                           {"generate": round(t_gen, 1), "bake": round(t_bake, 1)}, peak_rss_gb=rss,
                           extra=extra or None)
        print(f"Receipt: {rp}")
    print(f"\nTotal time: {time.time() - t0:.1f}s")
    return outputs


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(argv)

    if not os.path.exists(args.image):
        print(f"Error: {args.image} not found")
        sys.exit(1)

    cfg = apply_preset(args.preset, pipeline_type=args.pipeline_type, steps=args.steps,
                       texture_size=args.texture_size, no_texture=args.no_texture,
                       bake_faces=args.bake_faces, normal_map=args.normal_map, two_stage=args.two_stage,
                       single_process=args.single_process)
    latents_path = args.latents or f"{args.output}.latents.pt"

    print("=" * 60)
    print("TRELLIS.2 on Apple Silicon (Trellis-3)")
    print("=" * 60)
    print(f"preset={args.preset or '-'} pipeline={cfg['pipeline_type']} steps={steps_spec(args, cfg) or 'default'} "
          f"texture={'none' if cfg['no_texture'] else cfg['texture_size']} bake_faces={cfg['bake_faces']} "
          f"normal_map={cfg['normal_map']} max_tokens={args.max_tokens}")

    if args.attn_auto_gb is not None:
        os.environ["ATTN_AUTO_GB"] = str(args.attn_auto_gb)
    if args.attn_tile:
        os.environ["ATTN_TILE"] = args.attn_tile
        print(f"attn_tile={args.attn_tile}: tiled attention, score matrix bounded to H x {args.attn_tile.replace(',', ' x ')}")
    if args.attn_chunk:
        os.environ["ATTN_CHUNK"] = str(args.attn_chunk)
        print(f"attn_chunk={args.attn_chunk}: NOTE this can push PyTorch's MPS attention off its fused kernel "
              "onto the math path (materialises H x chunk x S scores). Use only if the unchunked run OOMs in attention.")
    profiler = None
    if args.profile_memory:
        from trellis3.profile import Profiler, activate
        profiler = Profiler().start("load")
        activate(profiler)

    t0 = time.time()

    # ---- decode-only process -------------------------------------------------
    if args.from_latents:
        state = load_state(args.from_latents)
        print(f"Latents: {args.from_latents} ({describe(state)})")
        if "tex" not in state.get("done", []):
            print("Error: latents are incomplete; run without --from-latents and with --resume to finish sampling")
            sys.exit(4)
        runner = Runner(state["pipeline_type"], resident=args.resident,
                        stage_hooks=not args.no_stage_hooks, decode_only=True)
        if profiler:
            profiler.mark("decode")
        try:
            mesh = runner.decode(state)
        except WatchdogError as e:
            print(str(e))
            sys.exit(2)
        t_gen = time.time() - t0
        print(f"\nMesh: {mesh.vertices.shape[0]:,} vertices, {mesh.faces.shape[0]:,} triangles")
        _export_and_finish(args, cfg, mesh, t0, t_gen, profiler, {"from_latents": args.from_latents})
        return

    # ---- sampling ------------------------------------------------------------
    runner = Runner(cfg["pipeline_type"], resident=args.resident, stage_hooks=not args.no_stage_hooks)
    img = PILImage.open(args.image)
    print(f"Input: {args.image} ({img.size[0]}x{img.size[1]})")
    if not args.no_thermal_guard:
        wait_until_cool()

    state = None
    if args.resume and os.path.exists(latents_path):
        state = load_state(latents_path)
        print(f"Resuming from {latents_path} ({describe(state)})")

    def on_stage(name, st):
        if not args.no_save_latents:
            save_state(latents_path, st)
            print(f"  saved latents after '{name}' -> {latents_path}")

    print(f"\nSampling (pipeline={cfg['pipeline_type']}, seed={args.seed})...")
    try:
        if args.legacy_run:
            mesh = runner.generate(img, seed=args.seed, pipeline_type=cfg["pipeline_type"],
                                   steps=cfg["steps"], max_num_tokens=args.max_tokens)
            t_gen = time.time() - t0
        else:
            state = runner.sample(img, pipeline_type=cfg["pipeline_type"], seed=args.seed,
                                  steps=steps_spec(args, cfg), max_num_tokens=args.max_tokens,
                                  state=state, on_stage=on_stage, release=not args.keep_models,
                                  min_free_gb=args.min_free_gb)
            t_sample = time.time() - t0
            print(f"Sampling time: {t_sample:.1f}s")

            if cfg["two_stage"]:
                if args.no_save_latents:
                    save_state(latents_path, state)
                child = [sys.executable, os.path.abspath(__file__)] + argv + ["--from-latents", latents_path]
                print(f"\nDecoding in a fresh process: {' '.join(child[2:])}")
                del runner
                memory.free_device_cache()
                code = subprocess.run(child).returncode
                sys.exit(code)

            if profiler:
                profiler.mark("decode")
            mesh = runner.decode(state)
            t_gen = time.time() - t0
    except MemoryPressure as e:
        print(f"\nStopped between steps: {e}")
        print(f"Completed stages are in {latents_path}. Resume with:\n"
              f"  python generate.py {' '.join(argv)} --resume")
        sys.exit(3)
    except WatchdogError as e:
        print(str(e))
        sys.exit(2)

    print(f"\nMesh: {mesh.vertices.shape[0]:,} vertices, {mesh.faces.shape[0]:,} triangles")
    print(f"Generation time: {t_gen:.1f}s")
    _export_and_finish(args, cfg, mesh, t0, t_gen, profiler, {"latents": None if args.no_save_latents else latents_path})


if __name__ == "__main__":
    main()
