#!/usr/bin/env bash
# Trellis-3 on an AMD GPU (ROCm) — Linux, WSL2, or Git-Bash on Windows with the
# ROCm-for-Windows PyTorch wheel. Mirrors setup.sh minus the Metal pieces.
#
#   ROCM_TORCH_INDEX  index URL for the torch wheel (default: ROCm 7.0 nightly).
#                     Windows ROCm users: set this to AMD's Windows wheel index.
#   HSA_OVERRIDE_GFX_VERSION  if ROCm does not list your GPU (Strix Halo: 11.5.1)
set -e
cd "$(dirname "$0")"
DEPS_DIR="deps"; mkdir -p "$DEPS_DIR"

[ -d TRELLIS.2 ] || git clone --depth 1 https://github.com/microsoft/TRELLIS.2.git TRELLIS.2
[ -d "$DEPS_DIR/utils3d" ] || git clone --depth 1 https://github.com/EasternJournalist/utils3d.git "$DEPS_DIR/utils3d"

if [ ! -d .venv ]; then
  if command -v uv >/dev/null; then uv venv .venv --python 3.11; else python3 -m venv .venv; fi
fi
# shellcheck disable=SC1091
source .venv/bin/activate 2>/dev/null || source .venv/Scripts/activate
PIP="pip install"; command -v uv >/dev/null && PIP="uv pip install"

ROCM_TORCH_INDEX="${ROCM_TORCH_INDEX:-https://download.pytorch.org/whl/rocm7.0}"
echo "torch from $ROCM_TORCH_INDEX"
$PIP --index-url "$ROCM_TORCH_INDEX" torch torchvision torchaudio
$PIP transformers accelerate huggingface_hub safetensors pillow numpy trimesh scipy tqdm easydict \
     kornia timm imageio opencv-python-headless xatlas fast-simplification psutil
$PIP "$DEPS_DIR/utils3d"

SKIP_METAL=1 python patches/mps_compat.py

cat <<MSG

=== ROCm setup complete ===
  source .venv/bin/activate      (Windows: .venv\\Scripts\\activate)
  export SPARSE_CONV_BACKEND=none ATTN_BACKEND=sdpa SPARSE_ATTN_BACKEND=sdpa
  python tools/check_rocm.py
  python generate.py photo.png --preset draft --device rocm
MSG
