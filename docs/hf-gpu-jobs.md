# Hero runs on a rented Hugging Face GPU (HF Jobs)

The laptop (Radeon 8060S, ROCm) takes about 13 minutes per sampling step at the hero setting (1536, 67k tokens, 4 views), roughly 5 hours
per head. On one A100-80GB the same head finishes in **61 minutes for about USD 2.53**. Upstream TRELLIS.2 is CUDA-native, so the job needs
none of the fork's ROCm/MPS patches; it runs this fork's `generate.py` (multi-image conditioning, staged sampling, receipt) with
`--device cuda` on top of an unpatched upstream clone.

## What was measured (vampire head, job `6ae85f9a`, 2026-11-02)

| item | value |
|---|---|
| flavor | `a100-large`: 1x A100 80 GB, 12 vCPU, 142 GB RAM, **$2.50/h** (billed per minute) |
| settings | `--preset hero --single-process --max-tokens 98304 --fusion multidiffusion --seed 42`, 4 views (front, three-quarter, back, profile; yaws 0 45 180 90), 1536 cascade, 67,150 tokens, 8K texture, 2M-face bake, normal map |
| attention | flash-attn 2.7.3 (fused), sparse conv `flex_gemm`, bake via official `o_voxel` on CUDA |
| wall | 60.7 min job (10:02:02 to 11:02:41 UTC); `generate.py` 3491 s |
| cost | 60.7 min x $2.50/h = **$2.53** |
| peak GPU | 48.1 GB reserved (tex decode); host RSS peak 48.1 GB (export) |
| stages | load 137 s, coords 9 s, shape_lr 11 s, shape_up 110 s, shape_hr 319 s (27 s/step), tex 303 s, shape decode 285 s (tiled), tex decode 33 s, export 2273 s |
| outputs | `mesh.glb` 387 MB (2M faces, 8K base colour/MR, 8K normal), `mesh_normal.png`, `mesh.receipt.json`, `job.log`, `mesh.latents.pt` (28 MB); `mesh.obj` (3 GB, the 70M-face decode) is left in the dataset repo |

Export (simplify 70M to 2M faces on CPU, UV unwrap, 8K bake, 8K normal map from the 70M-face mesh) is 38 of the 61 minutes and runs on a
mostly idle GPU. It is the part to optimise if cost matters (more vCPUs do not exist on `a100-large`; `h200` has 23 vCPU at $5/h).

Things that went wrong on the way, so you can skip them:

* First hero run OOM'd at decode with 79 GB used: the staged decode ran without `torch.no_grad` (46 GB wasted). Fixed on branch
  `memory-patches-188` (no_grad decode, free sampling tensors after decode, 256-voxel decoder tiling, latents kept and resumable). This
  branch contains that merge. Sampling latents are uploaded even when decode fails, so a failure costs a decode-only rerun
  (`--resume`), not a resample.
* `bash -lc` as the job command: `hf jobs run` parses `-l` as its label option. Use `bash -c`.
* `pip install` of the four CUDA wheels together fails resolution: install them with `--no-deps`.
* `transformers` must be 4.56.x (upstream's DINOv3 wrapper uses `.layer`; 5.x removed it).
* o-voxel needs Eigen: clone TRELLIS.2 with `--recursive`.
* The cut-out step is not in the job: inputs must be real RGBA cut-outs (`TRELLIS3_REMBG=none`). Fully opaque "RGBA" is treated as no alpha and aborts.

## Recipe

Files in `tools/hf_job/`: `build_wheels.sh` (CPU job), `run_hero.sh` (GPU job), `submit.py` (packs the repo with `git archive`, uploads
bundle + inputs to a private dataset repo, starts the job). Everything moves through the private dataset repo `Domlynch/trellis-hero-io`
(`wheels/`, `bundle/`, `runs/<run>/in|out`). The 16 GB TRELLIS.2 weights and the gated DINOv3 weights are pulled inside the job from the Hub;
nothing large is uploaded from the laptop. The HF token is forwarded as a job secret (`-s HF_TOKEN`, name only) and never printed.

```bash
export PATH="$PATH:$HOME/AppData/Local/Programs/Python/Python312/Scripts" PYTHONUTF8=1
hf auth whoami                                  # Domlynch, PRO, token needs the jobs scope
hf jobs hardware                                # flavors and prices

# 1. once (about 15 min on cpu-xl, about $0.25): compile nvdiffrast, CuMesh, FlexGEMM, o_voxel for sm80/sm90 into wheels
python tools/hf_job/submit.py wheels --go

# 2. a hero run: views are named view_0.png ... view_N.png, front first, RGBA cut-outs
python tools/hf_job/submit.py hero --run head1 --flavor a100-large --timeout 2h \
  --inputs view_0.png view_1.png view_2.png view_3.png \
  --gen-args "--preset hero --single-process --max-tokens 98304 --fusion multidiffusion --seed 42 --yaws 0 45 180 90" --go

# 3. follow and fetch
hf jobs logs <job_id>                           # set PYTHONUTF8=1 on Windows or the log decode fails
python -c "from huggingface_hub import snapshot_download as s; s('Domlynch/trellis-hero-io', repo_type='dataset', allow_patterns=['runs/head1/out/*'], ignore_patterns=['*.obj'], local_dir='out')"
```

Drop `--go` to only print the `hf jobs run` command. `--ref` picks the git ref bundled (default `HEAD`); it must contain `generate.py --images`
and `tools/hf_job`. Run the 12-angle turntable (`tools/turntable.py`, Blender on CPU) and the materials/projection passes on the laptop
after download; the job does not run them.

The laptop's DNS drops intermittently (`getaddrinfo failed`); logs and downloads simply need retrying.

## Cheap plumbing check

`--flavor a10g-small --gen-args "--pipeline-type 512 --steps 4 --texture-size 512 --bake-faces 50000 --single-process"` with two small cut-outs
runs end to end in about 5 minutes for about $0.10 (also installs, downloads the weights and exercises the multi-view path).

## Cost estimate per character

Measured: one hero head, 4 views, **$2.5 (61 min)**. Estimates below are extrapolations, not measurements; they scale with token count
(sampling) and with the bake (fixed 2M faces / 8K):

| job | estimate |
|---|---|
| hero head (measured) | $2.5 |
| hero part with fewer tokens (hands, feet, props) | $1.5 to 2.5, bake dominates |
| 1024 cascade part (body, coat) | about $1 to 1.5 |
| failed run that resumes from latents (decode + bake only) | about $1.7 |
| a character with head + body + coat + 4 small parts | about $8 to 10 |

Add about $0.1 per cold start for installs and weights (about 5 min, included in the 61 above). Wheels are built once. Budget rule used so far:
one GPU job at a time, a cap in USD agreed with the owner, no new paid submit without a yes.

## Licences (reported, not decided)

* TRELLIS.2 code and `microsoft/TRELLIS.2-4B` weights: MIT. `microsoft/TRELLIS-image-large` (sparse-structure decoder): MIT.
* DINOv3 `facebook/dinov3-vitl16-pretrain-lvd1689m`: Meta's DINOv3 License, gated (manual approval); the account has access and the job reads it with
  the forwarded token. Read its terms before shipping.
* `briaai/RMBG-2.0`: CC BY-NC 4.0, non-commercial; **not used** (`TRELLIS3_REMBG=none`, inputs are pre-cut RGBA).
* BiRefNet (`ZhengPeng7/BiRefNet`, used for cut-outs on the laptop): MIT.
* Pixal3D `TencentARC/Pixal3D`: MIT (Tencent responsible-use notice). Not run on HF Jobs yet; it is CUDA-native, so the same wheels and
  image would run it with `tools/pixal3d_entry.py`, but that path is untested here.
* Build tooling, NOT verified this session (the licence files could not be fetched): nvdiffrast 0.4.0 is, as far as I recall, under NVIDIA's
  source-code licence with non-commercial terms, CuMesh and FlexGEMM are from the TRELLIS.2 author's repos (check their LICENSE files),
  flash-attention is BSD-3. Check nvdiffrast before shipping anything that bundles it; it is a rendering dependency, not part of the mesh.
* Hugging Face Jobs: paid usage billed to the account (PRO).
