#!/usr/bin/env bash
# GPU job: install wheels, fetch the code bundle + inputs, run the hero head, push results to the private dataset repo.
# Image: pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel
# env: IO_REPO  RUN_NAME  GEN_ARGS (generate flags)   job secret: the HF credential (gated DINOv3 needs it)
set -uxo pipefail
T0=$(date +%s)
apt-get update -qq && apt-get install -y -qq git ninja-build libjpeg-dev libgl1 libglib2.0-0 >/dev/null
pip install -q huggingface_hub hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1
W=/work; mkdir -p $W/out && cd $W
python -c "
import os; from huggingface_hub import snapshot_download
snapshot_download(os.environ['IO_REPO'], repo_type='dataset', local_dir='$W/io', allow_patterns=['wheels/*','bundle/*','runs/'+os.environ['RUN_NAME']+'/in/*'])"
pip install -q --no-deps io/wheels/*.whl
pip install -q imageio imageio-ffmpeg tqdm easydict opencv-python-headless trimesh transformers accelerate safetensors \
  scipy kornia timm xatlas fast-simplification pandas lpips zstandard psutil pillow numpy
pip install -q git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8
pip install -q https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl || echo "flash-attn wheel failed, falling back to sdpa"
mkdir -p repo && tar -xzf io/bundle/trellis3.tar.gz -C repo && cd repo
git clone -q https://github.com/microsoft/TRELLIS.2.git TRELLIS.2
git -C TRELLIS.2 checkout -q "$(cat $W/io/wheels/trellis2.commit)"
# only the rembg-choice patch of the fork (RMBG-2.0 is non-commercial and gated); the MPS/ROCm patches stay off
python -c "import patches.mps_compat as m; m.patch_rembg_choice()"
export TRELLIS3_REMBG=none ATTN_BACKEND=flash_attn SPARSE_ATTN_BACKEND=flash_attn SPARSE_CONV_BACKEND=flex_gemm
python -c "import flash_attn" 2>/dev/null || export ATTN_BACKEND=sdpa SPARSE_ATTN_BACKEND=sdpa
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OPENCV_IO_ENABLE_OPENEXR=1 PYTHONUNBUFFERED=1
nvidia-smi --query-gpu=name,memory.total --format=csv
IN=$W/io/runs/${RUN_NAME}/in
IMGS=$(cd $IN && ls view_*.png | sort | sed "s|^|$IN/|" | tr '\n' ' ')
python $W/repo/generate.py --images $IMGS --device cuda --no-thermal-guard --profile-memory --output $W/out/mesh $GEN_ARGS 2>&1 | tee $W/out/job.log
RC=${PIPESTATUS[0]}
echo "generate exit code $RC; wall $(( $(date +%s) - T0 )) s" | tee -a $W/out/job.log
python - <<'P'
import os
from huggingface_hub import HfApi
run = os.environ["RUN_NAME"]
HfApi().upload_folder(folder_path="/work/out", path_in_repo=f"runs/{run}/out", repo_id=os.environ["IO_REPO"], repo_type="dataset",
                      ignore_patterns=["*.latents.pt"])
P
exit $RC
