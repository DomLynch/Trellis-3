#!/usr/bin/env bash
# Trellis-3 on an AMD GPU (ROCm). Two modes:
#
#   bash setup_rocm.sh            # "ports": AMD's ROCm 10 PyTorch + the ROCm builds of
#                                 #   flex_gemm / cumesh / o_voxel / nvdiffrast from
#                                 #   ComfyUI-Trellis2-AMD (fat wheels incl. gfx1150-1153 =
#                                 #   Ryzen AI Max+ 395) + aule flash attention. This is the
#                                 #   official GPU code path, end to end. Needs Python 3.12.
#   ROCM_PORTS=0 bash setup_rocm.sh   # "pure": any ROCm torch wheel + pure-PyTorch/CPU fallbacks
#
#   GFX=gfx1151           GPU target for the AMD torch index (Strix Halo default)
#   ROCM_TORCH_INDEX=...  (pure mode only) wheel index, default pytorch.org rocm7.0
#   HSA_OVERRIDE_GFX_VERSION=11.5.1   if ROCm does not list your GPU at runtime
#
# Linux (Ubuntu 24.04 recommended by the wheel author), WSL2, or Git-Bash on Windows.
set -e
cd "$(dirname "$0")"
DEPS_DIR="deps"; mkdir -p "$DEPS_DIR"
PORTS="${ROCM_PORTS:-1}"
GFX="${GFX:-gfx1151}"
PY="${PYTHON_VERSION:-3.12}"

[ -d TRELLIS.2 ] || git clone --depth 1 https://github.com/microsoft/TRELLIS.2.git TRELLIS.2
[ -d "$DEPS_DIR/utils3d" ] || git clone --depth 1 https://github.com/EasternJournalist/utils3d.git "$DEPS_DIR/utils3d"

if [ ! -d .venv ]; then
  if command -v uv >/dev/null; then uv venv .venv --python "$PY"; else "python$PY" -m venv .venv || python3 -m venv .venv; fi
fi
# shellcheck disable=SC1091
source .venv/bin/activate 2>/dev/null || source .venv/Scripts/activate
PIP="pip install"; command -v uv >/dev/null && PIP="uv pip install"
PYV=$(python -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')
PYTAG=$(python -c 'import sys; print(f"cp{sys.version_info[0]}{sys.version_info[1]}")')

if [ "$PORTS" = "1" ]; then
  echo "== ROCm 10 PyTorch stack for $GFX (AMD index) =="
  $PIP --extra-index-url https://stable.repo.amd.com/rocm/whl-next/ \
    "torch[device-$GFX]==2.13.0+rocm10.0.0" "torchvision[device-$GFX]==0.28.0+rocm10.0.0" \
    "torchaudio==2.11.0.2+rocm10.0.0" "rocm[libraries,device-$GFX]==10.0.0"
else
  ROCM_TORCH_INDEX="${ROCM_TORCH_INDEX:-https://download.pytorch.org/whl/rocm7.0}"
  echo "== torch from $ROCM_TORCH_INDEX =="
  $PIP --index-url "$ROCM_TORCH_INDEX" torch torchvision torchaudio
fi

$PIP "transformers==4.56.0" accelerate huggingface_hub safetensors pillow numpy trimesh scipy tqdm easydict \
     kornia timm imageio opencv-python-headless xatlas fast-simplification psutil
$PIP "$DEPS_DIR/utils3d"

if [ "$PORTS" = "1" ]; then
  echo "== ROCm builds of flex_gemm / cumesh / o_voxel / nvdiffrast (ComfyUI-Trellis2-AMD) =="
  [ -d "$DEPS_DIR/trellis2-amd" ] || git clone --depth 1 https://github.com/dmonkman/ComfyUI-Trellis2-AMD.git "$DEPS_DIR/trellis2-amd"
  case "$(uname -s)" in MINGW*|MSYS*|CYGWIN*|Windows*) OSDIR=Windows;; *) OSDIR=Linux;; esac
  WHL="$DEPS_DIR/trellis2-amd/wheels/$OSDIR/Python$PYV"
  ls "$WHL"/*.whl >/dev/null 2>&1 || { echo "no wheels for $OSDIR / Python $PYV in $WHL (use Python 3.12)"; exit 1; }
  $PIP --force-reinstall --no-deps "$WHL"/cumesh-*"$PYTAG"*.whl "$WHL"/flex_gemm-*"$PYTAG"*.whl \
       "$WHL"/o_voxel-*"$PYTAG"*.whl "$WHL"/nvdiffrast-*"$PYTAG"*.whl
  $PIP aule_attention || echo "  aule_attention install failed — attention falls back to tiled SDPA"
  [ "$OSDIR" = "Windows" ] && { $PIP triton-windows || echo "  triton-windows failed — flex_gemm needs it on Windows"; }
fi

SKIP_METAL=1 python patches/mps_compat.py

cat <<MSG

=== ROCm setup complete (ports=$PORTS) ===
  source .venv/bin/activate      (Windows: .venv\\Scripts\\activate)
MSG
if [ "$PORTS" = "1" ]; then cat <<MSG
  export SPARSE_CONV_BACKEND=flex_gemm ATTN_BACKEND=sdpa SPARSE_ATTN_BACKEND=sdpa ATTN_IMPL=aule TRELLIS3_OVOXEL=native
MSG
else cat <<MSG
  export SPARSE_CONV_BACKEND=none ATTN_BACKEND=sdpa SPARSE_ATTN_BACKEND=sdpa
MSG
fi
cat <<MSG
  python tools/check_rocm.py
  python generate.py photo.png --preset draft --device rocm
MSG
