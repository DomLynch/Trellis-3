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
