# Trellis-3 — TRELLIS.2 on Apple Silicon, tuned for 24GB machines

A fork of [shivampkumar/trellis-mac](https://github.com/shivampkumar/trellis-mac)
(MIT) that targets the things that actually hurt on a MacBook Air / 24GB Mac:
swapping, re-loading 15GB of weights for every try, and the small holes the
Mac port leaves in every mesh.

Upstream does the hard part — running Microsoft's CUDA-only
[TRELLIS.2](https://github.com/microsoft/TRELLIS.2) on Metal. This fork
does not touch the Metal kernels. It changes what gets loaded, when memory is
released, how jobs are run, and what happens to the mesh after decoding.

## What changed

| | Upstream trellis-mac | Trellis-3 |
|---|---|---|
| Checkpoints loaded | all 8 (~15 GB) | only those the pipeline type needs: 6 for `512` (~9.7 GB), 7 for `1024_cascade` (~12.3 GB) |
| MPS allocator | never flushed | `torch.mps.empty_cache()` after every sampling/decoding stage |
| Weight loading | ~100 s per image | `serve.py` loads once; watch-folder + HTTP, seed sweeps in one call |
| Hole filling | disabled (Metal cumesh segfaults) | CPU implementation, same `max_hole_perimeter` semantics as cumesh |
| `remove_faces` / `simplify` | disabled | CPU implementations |
| Presets | — | `--preset draft` (512, 8 steps, no texture) / `--preset final` |
| Cascade token budget | fixed 49152 | `--max-tokens` — the main memory lever at 1024/1536 |
| Thermal | — | waits for `pmset` `CPU_Speed_Limit` to recover before a job |
| Metal stack check | silent fallbacks | `tools/check_metal.py` says what is actually active |
| `1536_cascade`, 4096 textures | not selectable | selectable; need far more memory than 24 GB (see below) |

Weight sizes are the bf16/fp16 checkpoints on Hugging Face. The 1024 cascade
never uses `tex_slat_flow_model_512`; the 512 pipeline never uses either
1024 flow model. Upstream's `low_vram=True` already shuttles each model
CPU→MPS per stage, so "stage offloading" was never the missing piece —
loading unused weights was.

## Status — read this before trusting the numbers

Developed and unit-tested on Linux without an Apple GPU. The pure-Python
parts (hole filling, face removal, simplification, the patch script, the
server, presets, thermal parser, CLI wiring) are covered by 54 tests that
run on CPU. **The end-to-end Metal pipeline has not been run by this fork's
author.** See [AUDIT.md](AUDIT.md) for exactly what was verified and what
needs a run on a real Mac. Expected memory savings are arithmetic on
checkpoint sizes, not measurements.

## Quick start

Same as upstream; `setup.sh` applies the extra patch and ends with the Metal
check.

```bash
git clone https://github.com/DomLynch/Trellis-3.git
cd Trellis-3
xcodebuild -downloadComponent MetalToolchain   # recommended
hf auth login                                   # gated: facebook/dinov3-vitl16-pretrain-lvd1689m, briaai/RMBG-2.0
bash setup.sh
source .venv/bin/activate
python tools/check_metal.py                     # all OK? then:
python generate.py photo.png --preset draft     # ~1 min geometry check
python generate.py photo.png --preset final --output keeper
```

**Existing trellis-mac install?** Copy `backends/`, `patches/`, `trellis3/`,
`generate.py`, `serve.py`, `tools/` over your checkout and run
`python patches/mps_compat.py` again. The patch detects upstream's `return`
stubs and replaces them; it is idempotent.

## Resident server

```bash
# watch a folder
python serve.py --watch inbox --out outbox
# and/or HTTP
python serve.py --http 127.0.0.1:8765 --out outbox
curl -X POST localhost:8765/generate -d '{"image": "/abs/path/cat.png", "seeds": [1,2,3,4,5], "preset": "draft"}'
curl localhost:8765/job/<job_id>
```

Sidecar `inbox/cat.json` next to `inbox/cat.png` takes the same keys:
`seed`, `seeds`, `preset`, `pipeline_type`, `steps`, `texture_size`,
`no_texture`, `max_tokens`. Output goes to `outbox/cat/seed<N>.glb|.obj`.
`--pipeline-types 512,1024_cascade` preloads both; otherwise a new type
loads lazily on first use and stays loaded.

## generate.py options

```
--preset draft|final      draft: 512, 8 steps, no texture. final: 1024_cascade, 12 steps, 2048 texture
--pipeline-type           512 | 1024 | 1024_cascade | 1536_cascade
--texture-size            512 | 1024 | 2048 | 4096
--steps N                 sampler steps for all three flow phases
--max-tokens N            cascade token budget (default 49152); lower = less memory, lower resolution
--resident                keep models on MPS instead of CPU shuttling (opt-in, unmeasured)
--no-stage-hooks          don't flush the MPS cache between stages
--no-thermal-guard        don't wait for the CPU to cool
```

Explicit flags override the preset.

## Memory guidance for 24 GB

- `512`: should drop from upstream's ~18–20 GB peak to roughly 13–15 GB
  (5.2 GB of unused 1024 weights no longer loaded). Not yet measured.
- `1024_cascade`: upstream 28–31 GB (swapping). This fork removes 2.6 GB of
  weights; the rest is activations. Use `--max-tokens 32768` or lower to
  trade resolution for memory. Still likely to swap on 24 GB.
- `1536_cascade`: about 3.4× the voxels of 1024. Needs 64 GB+ in practice.
- 4096 textures: 4× the texels of 2048 through the Metal baker. Untested.

## Hole filling semantics

Meshes are in [-0.5, 0.5]³. `fill_holes(max_hole_perimeter=3e-2)` closes
only boundary loops shorter than 0.03 — tiny decode gaps, never an
intentional opening. Each loop is closed with a centroid fan wound to match
the surrounding surface. Loops that share a vertex (non-manifold boundary)
are left alone rather than guessed. 327K-face mesh with 10 holes: 0.2 s.

## Credits

- [TRELLIS.2](https://github.com/microsoft/TRELLIS.2) — Microsoft Research
- [trellis-mac](https://github.com/shivampkumar/trellis-mac) — Shivam Kumar; the Metal port this builds on
- [@pedronaugusto](https://github.com/pedronaugusto) — mtlgemm, mtldiffrast, mtlbvh, mtlmesh, o_voxel Apple fork

Upstream README (unchanged, for the port internals and benchmarks):
[docs/UPSTREAM_README.md](docs/UPSTREAM_README.md).

## License

MIT, same as upstream. Model weights carry their own licenses (TRELLIS.2 MIT;
DINOv3 Meta custom, gated; RMBG-2.0 CC BY-NC 4.0, non-commercial).
