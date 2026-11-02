# Memory patches for TRELLIS.2 issue 188 (branch memory-patches-188)

Source: microsoft/TRELLIS.2 issue 188, "OOM on smaller GPUs", opened 2026-09-07 by **rpep** (sampling tensors are still resident when `decode_shape_slat` / `Mesh.fill_holes()` run: peak ~21 GB drops to ~11 GB when they are freed first), with the tiling and export findings of **KaiStarkk** (comment of 2026-10-14, tested at TRELLIS.2 `75fbf01`). A public patch set is linked from that comment; it is under a non-standard licence ("MIT-SLOP"), so **nothing here was copied or vendored from it**: everything below was written from the issue text and from reading the decoder and `to_glb` of the installed TRELLIS.2 / o_voxel (MIT).

| Change | Where | Switch | Output |
|---|---|---|---|
| Latents are really freed before hole filling: `decode_state` hands ownership of the shape and texture latents to the decode (it used to keep them referenced in its own frame), moves the state's device tensors (coords, latents) to the CPU first, empties the cache | `trellis3/stages.py` (`_decode_lean`, `_offload_state`) | always on in the lean decode | identical mesh |
| Attribute volume (voxel colours) and its coords stay on the CPU until the bake | `stages.py` (`attrs_to_cpu`), `export.py` | `attrs_to_cpu=False` | identical |
| Export: the volume reaches the GPU only for the bake; the old CuMesh is released before a remesh; (the installed cumesh already chunks `remesh_narrow_band_dc`) | `trellis3/glb_lean.py` (guarded source edits of the installed `o_voxel.postprocess.to_glb`; an edit whose anchor is missing is skipped and reported in `paths["glb_lean"]`) | `TRELLIS3_GLB_LEAN=0` | identical (the fork's export does not remesh, so that edit only matters with `remesh=True`) |
| Shape decoder, last two stages in spatial tiles: cores of `tile_voxels` per side at the stage-3 resolution, margin = max(8, computed reach of the stages' 3x3x3 convolutions), stitched back in the original row order (also the subdivision tensor the texture decoder is guided by) | `trellis3/decoder_tiling.py` | `TRELLIS3_TILE_VOXELS=N` or `decoder_tiling.install(decoder, N)`; **0 = off, the default** | one tile = bit-identical; real tiles differ by float noise only (issue: 0.016 largest feature difference at 1024 with 128-voxel tiles) |

## Tests (CPU, `SPARSE_CONV_BACKEND=none`, real `SparseUnetVaeDecoder`)

`tests/test_decoder_tiling.py`: tile off vs one tile: equal feats, coords and all four subdivision tensors; 4x4x4 tiles and a two-batch 8-voxel run: same rows in the same order, max difference <= 1e-4; computed margin. `tests/test_lean_decode.py`: neither latent is alive when `fill_holes` runs, attribute volume on the CPU. `tests/test_glb_lean.py`: the three source edits on a synthetic source and on the installed `to_glb`.

## Measuring on the GPU

`python tools/memcheck_188.py LATENTS --mode baseline` then `--mode lean [--tile 256]` (one mode per process; GPU free, AC power). Records `torch.cuda.max_memory_allocated` for the decode (and with `--export-texture-size N` for the export), vertex and face counts and a sha256 of the positions. Acceptance: counts and `positions_sha256` equal between baseline and lean.
