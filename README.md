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
| Metallic | Metal path exports `metallicFactor=1.0` × model metallic (chrome skin) | `--metallic-factor` pinned to 0.0 by default |
| Base colour | Metal path writes linear (dark); KDTree path applies 1/2.2 | `--gamma 2.2` to match the two paths |
| Face winding | Metal cumesh `unify_face_orientations` left ~half flipped | consistent outward winding pass on the simplified mesh before bake (default on) |
| UV charts | 11–21K tiny charts ("confetti") | `--uv-cone-deg` / `--uv-smooth` passed to the chart clusterer (upstream 90 / 1) |
| Run receipt | — | `<output>.receipt.json`: settings, input hash/size, output hashes, timings, peak RSS, versions |
| Server memory guard | — | `serve.py --min-free-gb N` waits for free RAM before each job, fails the job after `--mem-wait` |
| Full-res detail in the bake | mtlbvh can't take the 800K-face decoder mesh, so the bake only ever sees the 200K simplified copy | CPU nearest-surface index on the full mesh: texels snap onto it before colour sampling (KDTree path) and `--normal-map` bakes a tangent-space normal map from it (both paths) |
| Cascade shape stage | one upstream call: 512 pass → decoder upsample → 1024 pass; nothing saved until all three finish | three sub-stages (`shape_lr`, `shape_up`, `shape_hr`), each saved to the latents file and each its own profiler row; `--resume` restarts at the one that died |
| Sampling vs decode | one `pipeline.run()` call: a decode crash loses 20 min of sampling, and the five flow models are still resident when the decoder starts | staged sampling with latents saved after every stage (`<output>.latents.pt`); each model dropped after its stage; `--two-stage` decodes in a fresh child process that loads only the two decoders (~1.9 GB); `--from-latents` / `--resume` |
| Memory guard | killed from outside, output lost | `--min-free-gb N` checks free RAM before every sampler step and exits cleanly (code 3) with the finished stages on disk |
| Steps | one `--steps` for all stages | `--ss-steps / --shape-steps / --tex-steps` (the texture stage is the slow one at 1024) |
| Bake size | 200K faces hard-coded | `--bake-faces N`; `--preset master` = 1024, 4096 textures, 1M faces, normal map, two-stage |
| 1024 attention on MPS | `scaled_dot_product_attention` materialises the full score matrix for these shapes: 12 × 17,730² × 4 B ≈ 15 GB — measured as the kill point of the 1024 shape pass on a 24 GB Mac | `--attn-tile 2048,4096`: exact tiled attention (online softmax, fp32 accumulators) with the score tile bounded to ~0.4 GB; slower, but it fits |
| 1024 attention memory | one SDPA call over all ~49K tokens, plus a padded copy of Q/K/V even for a single sequence | `--attn-chunk N` splits queries into chunks (exact; bounds the score matrix to H×N×S); uniform batches are zero-copy views |
| Coordinate lookups | Python dict of tuples + `.item()` loops in mesh extraction and the naive sparse conv (minutes and GBs at 1024) | sorted int64 keys + `searchsorted`: 3.6M voxels / 14M queries in 2.3 s |
| Memory attribution | — | `--profile-memory`: per-stage peak RSS / MPS driver / MPS allocator, printed and stored in the receipt |
| Colour match | `--gamma 2.2` overshoots (measured: 2.2 gives 0.53/0.45/0.40 vs photo 0.40/0.28/0.24) | `--gamma auto` solves the gamma that matches the texture's mean to the input photo; `--gamma 1.7` if you want it fixed |

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

**After every `git pull`:** re-run `python patches/mps_compat.py` — it re-copies
the helper modules (attention, lookups, mesh ops) into the TRELLIS.2 tree.

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
loads lazily on first use and stays loaded. Bake keys (`metallic_factor`,
`gamma`, `fix_winding`, `uv_cone_deg`, `uv_smooth`) are accepted per job.
`--min-free-gb 6` makes the server wait (up to `--mem-wait` seconds) for
free RAM before starting a job instead of swapping. Every output gets a
`.receipt.json`.

## Desktop master → phone

```bash
# master: 1024 shape, 4K textures, 1M-face bake, normal map; decode in a fresh process
# --attn-tile is what makes the 1024 shape pass fit on a 24 GB Mac
python generate.py head.png --preset master --output boss1_master --min-free-gb 3 --attn-tile 2048,4096
# it died at decode? sampling is on disk — just decode again
python generate.py head.png --from-latents boss1_master.latents.pt --preset master --output boss1_master
# it died during sampling? finished stages are kept
python generate.py head.png --preset master --output boss1_master --resume
# phone version: same latents, cheaper bake
python generate.py head.png --from-latents boss1_master.latents.pt --output boss1_phone --texture-size 1024 --bake-faces 30000
```

The latents file (tens of MB) can also be decoded on another machine.

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
--attn-tile LQ,SK         exact tiled attention, memory bounded (use 2048,4096 for 1024 on 24 GB)
--attn-chunk N            query-chunked full attention; exact. CAUTION: can force MPS off its fused kernel and use
                          MORE memory — only if the unchunked run OOMs inside a sampling pass
--profile-memory          per-stage memory table (RSS, MPS driver, MPS allocator) + receipt entry
--metallic-factor F       pin glTF metallicFactor (default 0.0; -1 keeps upstream's 1.0)
--gamma auto|G|0          base-colour gamma: auto matches the photo (default, ≈1.6–1.7 on heads), a number, or 0 for none
--bake-faces N            baked/exported face count (default 200000; master 1000000)
--ss-steps/--shape-steps/--tex-steps N   per-stage sampler steps
--two-stage               sample here, decode + export in a child process
--from-latents PATH       decode saved latents (loads decoders only)
--resume                  continue from <output>.latents.pt
--min-free-gb N           stop cleanly between steps when free RAM < N
--keep-models             do not drop models after their stage
--legacy-run              upstream pipeline.run() in one call
--normal-map              bake a normal map from the full-res decoder mesh (CPU, ~20 s at 2048)
--normal-flip-v           flip V for the normal map if it looks vertically wrong (Metal UV convention unverified)
--no-snap-hires           KDTree path: don't snap texels onto the full-res mesh
--no-fix-winding          skip the winding pass before the bake
--uv-cone-deg D           UV chart cone half-angle (upstream 90); larger = fewer charts
--uv-smooth S             UV chart smoothing (upstream 1); larger = fewer charts
--no-receipt              don't write <output>.receipt.json
```

Explicit flags override the preset.

## Memory guidance for 24 GB

- `512`: should drop from upstream's ~18–20 GB peak to roughly 13–15 GB
  (5.2 GB of unused 1024 weights no longer loaded). Not yet measured.
- `1024_cascade`: upstream 28–31 GB (swapping). This fork removes 2.6 GB of
  weights, the dict-based mesh extraction (GBs of Python objects at this
  size), and the padded Q/K/V copies; `--attn-chunk 4096` bounds attention
  if PyTorch's MPS kernel materialises the score matrix. Run once with
  `--profile-memory` — the stage table says what is left. `--max-tokens 32768`
  trades resolution for memory as a last resort.
- `1536_cascade`: about 3.4× the voxels of 1024. Needs 64 GB+ in practice.
- 4096 textures: 4× the texels of 2048 through the Metal baker. Untested.

## Measured on a 24 GB M5 Air (31 Oct 2026, Demon 2 head, pipeline 512)

| | upstream trellis-mac | Trellis-3 |
|---|---|---|
| Peak memory | 20.6 GB | **14.4 GB** (−30%) |
| Wall time | ~6 min | ~8 min (being investigated — see AUDIT) |
| Bake path | Metal | Metal |
| Winding | ~half the faces flipped | fixed by the winding pass |
| `--gamma 2.2` | — | too bright; 1.6–1.7 matches the photo → use `--gamma auto` |
| `--uv-cone-deg 120 --uv-smooth 2` | — | no visible change in chart count |

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
