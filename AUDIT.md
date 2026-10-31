# Audit — Trellis-3 initial release

Single self-audit done after the work, 31 Oct 2026. Environment: Linux
container, CPU-only PyTorch 2.15, no Apple GPU. Upstream TRELLIS.2 source was
cloned and used for the patch tests; no model weights were downloaded.

## Verified here

| Item | How | Result |
|---|---|---|
| CPU hole filling closes small loops, leaves large ones, keeps winding/normals | `tests/test_mesh_cleanup.py` (icosphere with 1–2 faces removed; trimesh `is_watertight`, `is_winding_consistent`, `volume > 0`) | pass |
| Default threshold matches cumesh scale (3e-2 in a [-0.5,0.5]³ mesh) | same file | pass |
| `remove_faces` compacts vertices; `simplify` reduces faces via fast_simplification | same file | pass |
| Torch wrappers preserve dtype (float32 / int32) and device | same file | pass |
| Patch applies to a **fresh** upstream `base.py` and to one **already stubbed** by trellis-mac; idempotent; fails loudly if a method is missing | `tests/test_patch.py` | pass |
| The **patched** `Mesh` class, imported for real, fills a hole / removes faces / simplifies | `tests/test_patch.py::test_patched_mesh_fills_hole_for_real` | pass |
| Full `patches/mps_compat.py` runs end-to-end on a clean TRELLIS.2 copy, twice | manual run | all 10 patches apply; second run reports "Already patched" for all |
| Only the needed checkpoints are requested via `model_names_to_load`, and the class attribute is restored | `tests/test_memory.py` with a fake pipeline | pass |
| Lazy loading of extra checkpoints in server mode; moved to device when `resident` | same | pass |
| Stage hooks fire after each stage, pass through return values, idempotent | same | pass |
| Presets and explicit-flag override rules | `tests/test_presets.py` | pass |
| `pmset` parsing; thermal guard never blocks without data; gives up at `max_wait` | `tests/test_thermal.py` | pass |
| Export (vertex-colour + OBJ path) writes valid files; empty mesh raises | `tests/test_export.py` | pass |
| Server: seed sweeps, failure isolation (one bad seed doesn't kill the server), lazy runner, inbox scan with sidecar, fresh-file skip, HTTP POST/GET round-trip on a live socket | `tests/test_serve.py` | pass |
| CLI wiring: preset → runner → export, `--max-tokens` reaches `pipeline.run` | `tests/test_generate_cli.py` | pass |
| Lint | pyflakes | clean except one intentional probe import |
| Hole filler speed at decoder scale | 327K-face icosphere | 0 holes 0.26 s; 10 holes 0.18 s; simplify 327K→200K 0.23 s |

54 tests, 2.7 s.

### Batch 2 (same day) — verified here

| Item | How | Result |
|---|---|---|
| `metallicFactor` pinned on Trimesh and Scene materials; `None` leaves it | `tests/test_postbake.py` | pass |
| Gamma curve brightens RGB, keeps alpha, no-op for None/1.0 | same | pass |
| Winding pass repairs a half-flipped icosphere to consistent + outward; no-op on a good mesh | same | pass |
| Metal-path wiring: winding fix runs before `to_glb`, UV kwargs / texture size reach it, metallic + gamma applied to its return value | `tests/test_export.py::test_metal_path_wiring` with a fake `o_voxel` | pass |
| Receipt JSON: settings, input size + sha256, output hashes, missing files skipped, timings, RSS, env, extra | `tests/test_receipt.py` | pass |
| `vm_stat` and `/proc/meminfo` parsers | `tests/test_memory.py` | pass |
| Memory guard waits, passes, gives up at `max_wait`, never blocks on unknown | `tests/test_serve.py` | pass |
| Server skips a job on low memory and keeps running; writes a receipt per seed | same | pass |
| CLI: bake flags reach `export_mesh`, receipt written | `tests/test_generate_cli.py` | pass |

68 tests, 2.5 s. Not verified: the real `to_glb` accepting the UV kwargs by
keyword (names were read from Pedro's `postprocess.py` signature, lines 88–91),
and whether `fix_normals` on a 200K-face decoder mesh is fast enough (trimesh
graph traversal; expect seconds, not minutes).

### Batch 3 — CPU nearest-surface ("BVH") and normal map

| Item | How | Result |
|---|---|---|
| Exact point-triangle closest point matches `trimesh.proximity.closest_point` (2000 random points, icosphere) | `tests/test_hires.py` | distance to 2e-5, point to 2e-4, barycentric reconstructs |
| Points on the surface report distance 0 and the right face | same | pass |
| Flat and smooth normals point outward; smooth is closer to the true sphere normal | same | pass |
| Nearest-vertex fast mode: error ≤ 0.6 × max edge on a dense sphere, normals within 0.98 of truth | same | pass |
| Normal map of sphere-on-sphere is flat blue (mean B > 235, R/G ≈ 127); bumped hi-res produces variation (std > 15) | same | pass |
| to_glb frame conversion inverted correctly | same | pass |
| Metal-path wiring: normal map baked from the full mesh, attached as `normalTexture`, PNG written | `tests/test_export.py::test_metal_path_normal_map` (fake to_glb that unwraps + converts frame) | pass |
| Benchmarks (Linux, 327K-face hi-res sphere) | manual | index 0.15 s; normal map 2048², 4.1M texels: **16.6 s** end to end; exact mode is ~100× slower and is opt-in |

81 tests, 3.6 s.

### Batch 4 — 1024 memory: attention, lookups, profiler

What the code actually does at 1024 (read from upstream, not guessed):
the SLat flow model is `attn_mode='full'` — every block attends over all
tokens (up to `max_num_tokens`=49152), 12 heads × 128, 30 blocks, bf16.
CFG runs cond and uncond **sequentially**, not batched. The decoder's
`low_vram` flag is set by the pipeline but consumed nowhere. The mesh
extraction and naive sparse conv built Python dicts of tuples with
per-element `.item()` loops.

| Item | How | Result |
|---|---|---|
| `pad_uniform` equals upstream loop padding (uniform and ragged); uniform case is a view, not a copy | `tests/test_attention_mps.py` | pass |
| `sdpa_chunked` equals full SDPA for chunk ∈ {1, 7, 49, 50, 500}; env parsing; end-to-end pad→chunk→unpad equals plain attention | same | pass (1e-6) |
| Attention patch applies to the upstream-inserted block (existing installs) and fresh installs; idempotent; compiles | `tests/test_patch.py` | pass |
| `CoordIndex` equals a dict oracle on random coords incl. misses, 4-D batch coords, out-of-range, empty | `tests/test_voxel_lookup.py` | pass |
| Vectorised mesh extraction produces the same quad set as the dict version; split-weight path; empty case | `tests/test_mesh_extract.py` | pass |
| Vectorised neighbour map equals a brute-force oracle for 3³ / dilated / anisotropic kernels and two batch items | `tests/test_conv_none.py` | pass |
| Profiler attributes peaks to stages; stage hooks mark it | `tests/test_profile.py` | pass |
| Full patch script on a clean TRELLIS.2 copy, twice; every patched/installed file compiles; flat-module import of the new mesh extraction works | manual | pass |
| Lookup benchmark, 1024-scale shell (3.57M voxels, 14.3M queries) | manual | **2.3 s** vectorised vs ~9 s for a `tolist()` dict (upstream used `.item()` per element — several × slower again) and ~1–2 GB of transient Python objects |

100 tests, 8.4 s.

Honest expectation for 1024: the certain wins are the lookups (time and
1–2 GB transient) and the padded-copy removal (~0.6 GB at 49K tokens).
Whether `--attn-chunk` saves anything depends on PyTorch's MPS SDPA kernel:
if it already streams over keys, chunking is neutral; if it materialises
[H, L, S], chunking is the difference between fitting and not. That is
exactly what `--profile-memory` is for — the `sample_shape_slat_cascade`
row will show it. Nothing here was run on MPS.

### Batch 5 — staged sampling, two-process decode, master preset

| Item | How | Result |
|---|---|---|
| Stage order and arguments equal upstream `run()` for `512` and `1024_cascade` (cond resolutions, ss_res, cascade args, max_num_tokens, per-stage steps) | `tests/test_stages.py` against a recording fake pipeline | pass |
| Models released after their stage (only the two decoders left), cond models dropped after conditioning | same | pass |
| Resume skips completed stages, reuses the preprocessed image, restores RNG, refuses a mismatched seed/type | same | pass |
| Memory guard checks before every sampler step, raises `MemoryPressure`, restores sampler methods, completed stages preserved | same | pass |
| `decode_state` rebuilds sparse tensors and calls `decode_latent`; rejects incomplete state | same | pass |
| Latents file round-trip (atomic write, CPU tensors, RNG, image bytes); foreign files rejected | `tests/test_latents.py` | pass |
| Decode-only load: just the two decoders, DINOv3/BiRefNet never constructed, module attributes restored | `tests/test_memory.py` | pass |
| CLI: staged run saves latents + receipt; `--preset master` spawns the child with the same args + `--from-latents`; decode-only path; incomplete latents exit 4; memory pressure exit 3 with partial latents; `--resume`; `--legacy-run` | `tests/test_generate_cli.py` | pass |

118 tests, 7.6 s.

Not verified on MPS: that `SparseTensor(feats, coords)` rebuilt from the
saved state decodes identically to the in-memory one (coords/feats are saved
verbatim, so it should), that `torch.mps.get/set_rng_state` round-trips, and
the actual memory headroom a decode-only child process has (expected ~20 GB
free on a 24 GB machine instead of ~8).

### Batch 6 — cascade split (from the Mac tester's crash analysis)

The 1024 run that was killed died *inside* the shape stage: after the 512
pass (44 s) and before the stage ended, so nothing of it was saved. The
cascade stage is now three sub-stages mirroring the upstream code line by
line (`_sample_slat`, `_upsample_coords` in `trellis3/stages.py`).

| Item | How | Result |
|---|---|---|
| Sub-stage order, saves and contents: `shape_lr` (512 latents), `shape_up` (hr coords + res), `shape_hr` (1024 latents); 1024 pass uses the 1024 model/cond/steps on the quantised coords | `tests/test_stages.py` | pass |
| Resume after `shape_lr` re-runs only the 1024 pass | same | pass |
| Token budget steps resolution down by 128 with a 1024 floor (upstream loop) | same | pass |
| 512 flow model released right after `shape_lr`; decoder shuttled back to CPU with `low_vram` reset | same | pass |
| Memory guard inside the cascade keeps `coords`, `shape_lr`, `shape_up` | same | pass |

120 tests.

**Where the kill most likely happened, and why chunking the upsample is not
the first move.** The upsample input is the 512 SLat (a few thousand tokens
on the 32³ structure grid); its output is just coordinates at 512 res
(~1M). Its activations at the finest level are N×64 fp16 — hundreds of MB,
not tens of GB. The thing that starts immediately after it is the **1024
sampling pass over up to 49K tokens with full attention**. Two facts from
that run: it had `--attn-chunk 4096` on, and the *previous* day's 1024 run
(upstream SDPA, no chunking) got through all of sampling. Chunking slices
Q so L≠S, which on MPS can drop SDPA from its fused kernel to the math
path that materialises `[12, 4096, 49152]` fp32 ≈ 9.6 GB per chunk plus
softmax temporaries — enough to collapse free memory right after the 512
pass. So the first retry is **without `--attn-chunk`**; the sub-stage
profile rows then say whether `shape_up` or `shape_hr` owns the peak. The
upsample gets chunked only if its row is the one that is high.

### Batch 7 — tiled attention (from the profile run)

Profile on `0fbf933`, master, head1, no `--attn-chunk`: `shape_lr` 51 s OK,
`shape_up` 9 s OK (17,730 tokens at 1024), **killed in `shape_hr`** with
free memory 54% → 14% in 20 s, swap 9.8 GB. That is the 1024 shape
sampling pass. At 17,730 tokens the attention score matrix is
12 × 17,730² × 4 B ≈ 15 GB, so PyTorch's MPS SDPA is materialising it on
this path. My earlier `--attn-chunk` only bounded the query axis and was
not the cause of the previous crash either (this run had it off).

| Item | How | Result |
|---|---|---|
| `sdpa_tiled` equals `scaled_dot_product_attention` for sizes that don't divide the tiles, single-row queries, S<L, and tiles larger than the input | `tests/test_attention_mps.py` | pass (1e-5) |
| bf16 inputs with large logits (running-max path), output dtype preserved | same | pass (3e-2, bf16 rounding) |
| Dispatcher: `ATTN_TILE` wins over `ATTN_CHUNK`; env parsing incl. `2048x4096` | same | pass |
| CPU timing, 12 × 8192 × 128 | manual | plain SDPA vs tiled — see commit message; exactness 1e-6 |
| Bound | arithmetic | tile `2048,4096`: 12 × 2048 × 4096 × 4 B = 0.4 GB per tile, vs 15 GB |

123 tests.

Not verified on MPS: the speed. It is a Python loop of (L/2048) × (S/4096)
matmuls per attention call — at 17.7K tokens 45 tiles × 30 blocks × 12
steps × 2 (CFG) ≈ 32K small kernels for the shape pass. Expect it slower
than the fused path would be, but it should run. If too slow, larger tiles
(`4096,8192` → 1.6 GB) trade memory for fewer launches.

`_metal_bake_available` now also accepts `_BACKEND == "cuda"`, so a latents
file produced on the Mac can be decoded on a CUDA machine with the official
bake.

### Batch 8 — 64 GB machine

| Item | How | Result |
|---|---|---|
| Automatic tiling only when `B·H·L·S·4` exceeds the budget; chosen tile stays under budget/4; disabled at 0 | `tests/test_attention_mps.py` | pass |
| `hero` preset; `--single-process` overrides a preset's two-stage | `tests/test_presets.py` | pass |

125 tests. Defaults unchanged for 24 GB users (the 6 GB auto budget tiles
the 1024 pass there, which is what the profile showed was needed).

### Batch 9 — ROCm plumbing (for the ROG Flow Z13)

| Item | How | Result |
|---|---|---|
| `--device` / `TRELLIS3_DEVICE`: auto order, explicit selection, loud failure when the backend is absent, `rocm` maps to `cuda` | `tests/test_memory.py` | pass |
| Profiler reads CUDA/ROCm allocator when present | code path; no GPU here | unverified |
| `setup_rocm.sh` parses; `tools/check_rocm.py` imports | `bash -n`, import | pass |

125 tests. Nothing ROCm-specific has been executed: no AMD GPU in this
container. Expected first-run issues on Strix Halo: wrong torch wheel for
gfx1151 (`check_rocm.py` says so), SDPA falling to the math path (auto-tiling
covers it), and `conv_none` speed. The two things that cannot be fixed from
here without a device: the sparse-conv speed (needs a ROCm `flex_gemm`) and
whatever the Windows ROCm wheel does differently from Linux.

### Batch 10 — trilinear CPU bake, machine-wide token budget

| Item | How | Result |
|---|---|---|
| `sparse_trilinear` equals `torch.grid_sample(mode='bilinear', align_corners=True)` on a dense 9³ volume at 500 random interior points | `tests/test_voxel_sample.py` | pass (1e-5) |
| Missing corners: renormalised by default, official zero-contribution with `renormalize=False`; points outside the grid report uncovered | same | pass |
| Chunking transparent | same | pass |
| `bake_texture(sampling="trilinear")` end to end on a quad over a uniform voxel cube: exact colour, roughness/metallic channels, gamma optional | same | pass |
| Full CPU bake path through `export_mesh`: xatlas unwrap → trilinear → metallic pinned → gamma auto from the photo → PNG + GLB | `tests/test_export.py::test_cpu_bake_path_trilinear_with_gamma_auto` | pass |
| Token budget: 98304 on ≥48 GB, 49152 below, explicit value wins, not tied to a preset | `tests/test_presets.py` | pass |
| Server routes `bake_faces`, `normal_map`, `max_tokens`, `bake_sampling` per job | `tests/test_serve.py` | pass |

132 tests.

Quality note: with trilinear sampling, snap-to-full-res, configurable bake
size, normal map and the gamma/metallic handling, the CPU bake now does the
same maths as the official `to_glb` at every step except UV unwrapping
(xatlas vs. Microsoft's own chart clustering). Not verified against a real
decoder output — the oracle here is `grid_sample`, which *is* the official
sampler, so the remaining risk is integration, not maths.

### First Mac measurements (reported by the Mac tester, 31 Oct 2026)

- Peak memory 14.4 GB vs 20.6 GB upstream, same method (−30%). Confirms the
  checkpoint-selection arithmetic within expectations (predicted 13–15).
- Metal bake path active; winding pass reported ~half the faces flipped and
  the fix works — this confirms the upstream `unify_face_orientations` is
  not effective on this install.
- Wall time rose from ~6 to ~8 min. Candidates: the winding pass
  (`fix_normals` on 200K faces — its time is printed in the log), the MPS
  cache flush forcing re-allocation each stage (`--no-stage-hooks` to A/B),
  and the thermal guard (`--no-thermal-guard`). Needs the per-stage times
  from the log to attribute.
- `--gamma 2.2` too bright (texture mean 0.53/0.45/0.40 vs photo
  0.40/0.28/0.24); 1.6–1.7 matches. So the Metal path's base colour is not
  plain linear either. `--gamma auto` now solves it from the photo.
- `--uv-cone-deg 120 --uv-smooth 2` did not visibly change chart count. The
  kwargs are accepted by `to_glb`, so either the clusterer ignores them in
  this path or the fragmentation comes from elsewhere (the pre-simplified
  mesh's connectivity, or flipped normals before the winding fix landed).
  Next: log the chart count and A/B cone 180 / smooth 10 vs default.

## Not verified — needs one run on a Mac

1. **End-to-end generation on MPS.** Nothing here ran the real pipeline. The
   code paths that touch it (`Runner`, `export_mesh` Metal branch, stage hooks
   wrapping real methods) were written against the upstream source and tested
   with fakes only.
2. **Memory numbers.** "~13–15 GB at 512" is checkpoint arithmetic
   (upstream ~18–20 GB minus 5.2 GB of unloaded 1024 weights, minus whatever
   `empty_cache()` reclaims). Measure with `Peak RSS` printed at the end of
   `generate.py`, before and after.
3. **`--resident`** (low_vram=False). Keeps selected models on MPS; avoids a
   2.6 GB copy per stage. Peak could be higher if MPS enforces a working-set
   limit. Off by default for that reason.
4. **`--max-tokens` effect on 1024 quality/memory.** The pipeline already
   lowers resolution when the budget is exceeded; the trade-off curve on a
   24 GB machine is unmeasured.
5. **4096 textures and `1536_cascade`.** Made selectable, expected to OOM on
   24 GB. Nothing in this fork makes them cheaper.
6a. **Normal map V convention on the Metal path.** The map is rasterised
   with the trimesh/glTF row convention (row = 1 − v) over the UVs `to_glb`
   returns. If it renders vertically mirrored, `--normal-flip-v`.
6b. **Rasteriser cost.** The normal map reuses upstream's per-face Python
   rasteriser, run twice (positions + face ids): ~10 s at 200K faces / 2048².
6c. **Nearest-vertex approximation.** Snap error ≤ half an edge of the
   decoder mesh (~0.001 at 512). Exact mode exists for validation.
6. **Hole filling on real decoder output.** Tests use synthetic holes. Real
   meshes may have boundary loops that share vertices (non-manifold
   boundary); those are skipped by design. In a 500-random-hole stress test
   494/500 were filled, the rest were adjacent-face cases left open.
   `fill_holes` runs on the full ~800K-face decode mesh; expect well under a
   second based on the 327K benchmark.
7. **Thermal guard on macOS.** `pmset -g therm` output format was taken from
   documentation; the parser is tested on that format only.
8. **setup.sh on macOS.** Only the appended `check_metal.py` line is new; the
   rest is upstream's.

## Known gaps / honest notes

- Earlier in the conversation that produced this fork, "stage offloading"
  was described as missing. It is not: upstream `low_vram=True` already
  moves models per stage. The real waste was loading unused checkpoints;
  that is what this fork fixes. The README says so.
- `fill_holes` uses a centroid fan. cumesh may use a different triangulation
  for larger loops; at the default 3e-2 perimeter the loops are a few edges
  and the difference is invisible.
- `simplify` ignores cumesh's `options` dict (fast_simplification has no
  equivalent). Upstream never calls `simplify` at decode time, so this is
  dormant.
- The pyflakes warning on `trellis3/runner.py` (`flex_gemm` imported but
  unused) is the intentional import-probe copied from upstream `generate.py`.
- Upstream `generate.py` behaviour is preserved except: output printing is
  slightly reorganised, `Peak RSS` is printed, and the thermal guard may
  pause before generation (disable with `--no-thermal-guard`).

## How to close the loop on a Mac

```bash
python tools/check_metal.py
python generate.py assets/shoe_input.png --preset draft           # geometry only, note Peak RSS
python generate.py assets/shoe_input.png --pipeline-type 512      # compare Peak RSS with upstream
python generate.py assets/shoe_input.png --pipeline-type 1024_cascade --max-tokens 32768
python serve.py --http 127.0.0.1:8765 --out outbox                # then POST a 5-seed job
```

If the first two run and the GLB opens with no holes where upstream had
them, items 1, 6 and 8 above are closed. The RSS lines close item 2.
