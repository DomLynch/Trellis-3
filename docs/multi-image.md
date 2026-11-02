# Multi-image input, projection texturing and Pixal3D

One image only ever shows one side of a character, so the back and sides of a single-image mesh are guessed and often come out speckled or
hollow. This branch adds three ways to give the model more views. Everything below is on `generate.py`; a single image with no new flag
takes exactly the old code path.

## 1. Multi-image conditioning (`--images`, `--fusion`)

Port of microsoft/TRELLIS.2 pull request 104, re-implemented in `trellis3/multiview.py` so the upstream clone stays untouched. Each view is
cut out and encoded separately; the per-view features are stacked and the sampler fuses them at every denoising step in all three stages
(sparse structure, shape, texture). No camera poses are needed.

```bash
python generate.py --images front.png three_quarter.png back.png profile.png --preset master --texture-size 2048 --bake-faces 500000 --attn-auto-gb 28 --device rocm --single-process --output out/head
```

| flag | meaning |
|---|---|
| `--images A B C ...` | views of the same subject, front first. `generate.py A.png --images B.png` also works (A is the first view) |
| `--fusion multidiffusion` (default) | every step averages the prediction of every view, then applies guidance once. N times the model calls, much more consistent |
| `--fusion stochastic` | step k uses view k mod N. Same cost as one image, each step sees one view |

Give cut-outs with an alpha channel and `TRELLIS3_REMBG=none` (the Factory's cut-out step does this). The first image is the main view. Order
only matters for `stochastic`. Latents, receipt and resume work as before; the receipt lists `images` and `fusion`.

## 2. Projection texturing (`--project`, `python -m trellis3.projection`)

Re-implementation of the idea in ComfyUI-Trellis2's "Projection MultiView Texturing" node, CPU only. After the bake, every texel of the base
colour texture is projected into each posed view. A view paints a texel if the texel is visible from it (depth test), faces it, and lies
inside the view's cut-out. Views are blended by those weights and the TRELLIS texel is kept where no view is confident (`--tau`, default 0.15), so
texels no view sees stay as generated.

```bash
python generate.py --images front.png three_quarter.png back.png profile.png --yaws 0 45 180 90 --project --preset master --texture-size 2048 --bake-faces 500000 --output out/head
# on an existing mesh:
python -m trellis3.projection out/head.glb out/head_proj.glb --view front.png:0 --view three_quarter.png:45 --view back.png:180 --view profile.png:90
```

Camera convention when poses are unknown: orthographic, elevation 0, yaw 0 front, 45 three-quarter, 90 profile (the camera is on the model's
+X side, so the face points to image-left), 180 back. Nominal poses are refined per view by fitting yaw (only if it clearly helps), scale and
shift to the mesh silhouette; the fit IoU is written to `<out>_proj.glb.projection.json`. Views must be RGBA cut-outs.

## 3. Pixal3D (`--preset pixal3d`)

TencentARC/Pixal3D (MIT) on the TRELLIS.2 backbone. It runs in its own venv; `trellis3/pixal3d.py` launches `tools/pixal3d_entry.py` there and
produces the same layout as a TRELLIS run: `<output>.glb`, `<output>_basecolor.png`, `_normal.png` / `_metallic_roughness.png` if present,
`<output>.receipt.json`.

```bash
python generate.py head_front.png --preset pixal3d --output out/head_px                                    # single view
python generate.py front.png three_quarter.png profile.png back.png --yaws 0 45 90 180 --preset pixal3d --output out/head_px_mv
```

Several images use Pixal3D's native multi-view model (`*_mv` weights, posed cameras written to a `transforms.json` next to the output). The
first image must be the front (yaw 0). The GLB is exported by the CPU port in `Pixal3D-run/pixal3d_glb_cpu.py` (no remesh) because cumesh
crashes on this ROCm build; results are labelled PX-CPUGLB. Paths default to this machine; override with `PIXAL3D_PY`, `PIXAL3D_REPO`,
`PIXAL3D_RUN`, `PIXAL3D_DINO`, `PIXAL3D_HF_HOME`, `PIXAL3D_TORCH_HOME`, `PIXAL3D_SHIMS`.

Licences (reported, not decided): Pixal3D code and weights MIT (Tencent, with a responsible-use notice); TRELLIS.2 MIT; DINOv3 Meta's DINOv3
License (its own terms, read it before shipping); BiRefNet MIT; MoGe-2 MIT; NAF Apache-2.0; RMBG-2.0 is non-commercial and is not used.

## Turntable sheet

`python tools/turntable.py mesh.glb [out_dir]` renders 12 yaw angles with Blender Cycles on the CPU, writes `contact.jpg` (4 x 3) and prints the
near-black texel share of the base colour (texels covered in UV space, luminance under 8).

## Tests

`tests/test_multiview.py` (conditioning tensor shapes, fusion maths, restore), `tests/test_projection.py` (view geometry, blend, depth test,
silhouette fit), `tests/test_pixal3d.py` (camera matrices against Pixal3D's example, view directory, command line, preset dispatch),
`tests/test_generate_cli.py` (`--images`, receipt keys, single-image path unchanged). Set `HIP_VISIBLE_DEVICES=-1` to keep them off the GPU.

## Power: no GPU runs on battery

On the ROG Flow Z13 a power-source change in the middle of a run (AC lost at 12:15, 2026-11-02) throttled the GPU from 86 to 140 s per step
and then aborted the HIP runtime (`c10::cuda::SetDevice`, python.exe fault 0xc0000409). `generate.py` therefore refuses to start a GPU run
(`--device rocm|cuda`, or `--preset pixal3d`) while the laptop is discharging and exits with code 5. Override with `--allow-battery` or
`TRELLIS3_ALLOW_BATTERY=1`. Keep the charger in and use a performance power plan for long runs.

## Measured on the 32 GB ROCm laptop (Radeon 8060S)

| attention | throughput | note |
|---|---|---|
| tiled fp32 (`--attn-auto-gb 12`) | 2.2 TFLOPs | exact, about 8x slower |
| fused SDPA flash (`--attn-auto-gb 0`) | 16-19 TFLOPs | exact, use this |

Hero (1536) shape sampling with 4 views and multidiffusion needs about 34 GB at 67k tokens, over the 32 GB dedicated share; the 1024
cascade runs at about 86 s per step. Pixal3D-MV loads about 22 GB of flow models: `tools/pixal3d_entry.py` loads them lazily (one at a time).

## Registered, front-owned projection (for faces)

Silhouette fitting can match an outline while features sit a centimetre off, and averaging misregistered views gives a ghosted face
(double moustache, extra eyes). For faces use landmark registration plus an owner blend:

```bash
python tools/face_register.py render mesh.glb reg --yaws 0 90                # exact-camera renders of the mesh, with grids
python tools/face_register.py crop cut_front.png reg/front_crop.png 1000 700 3100 3700 0.4 100    # zoomed grid crops to pick on
python tools/face_register.py solve reg/landmarks.json reg/solve               # cameras.json + overlays
python -m trellis3.projection mesh.glb mesh_proj2.glb --cameras reg/solve/cameras.json --size 8192 \
    --view front.png:0 --view tq.png:45:1.0 --view tq_mirror.png:-45:1.0 --view profile.png:90:0.8 --view profile_mirror.png:-90:0.8
python tools/unlit_closeups.py mesh_proj2.glb out/proj2 reg/solve/cameras.json --yaws 0 45    # unlit close-ups to judge it
```

Landmarks (picked by eye; `eye_vl`/`eye_vr` = viewer-left/right in the front view, `nose_base`, `mouth_c`, `chin`, `glabella`) are lifted to 3D
from the mesh render's depth and matched to the picks on each cut-out view. Each view gets an affine camera plus a thin-plate-spline warp so
every picked feature lands exactly (the photo face is wider than the mesh face, so a rigid camera cannot align them). Mirrored views reuse
the picks with left/right swapped. The first view owns every face texel within 55 degrees of its axis (fading out at 75), so eyes, brows, nose,
mouth and moustache come from the front only; the other views fill beyond that and the single best one wins. Hair, bust and the back of the head
stay the raw bake, and the output GLB drops its normal map. On the 1024 head this removed the ghosting of the averaged projection.

