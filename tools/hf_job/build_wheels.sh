#!/usr/bin/env bash
# CPU job (cpu-xl, ~$1/h): compile the CUDA extensions TRELLIS.2 needs into wheels and store them in the private
# HF dataset repo, so GPU jobs install in a minute instead of compiling at GPU prices.
# Image: pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel   env: IO_REPO=<user>/<dataset>   job secret: the HF credential
set -euxo pipefail
export TORCH_CUDA_ARCH_LIST="8.0;9.0"      # A100, H100/H200
export MAX_JOBS=16 FORCE_CUDA=1
apt-get update -qq && apt-get install -y -qq git ninja-build libjpeg-dev >/dev/null
pip install -q ninja setuptools wheel pybind11 huggingface_hub
W=/tmp/wheels; mkdir -p $W /tmp/ext; cd /tmp/ext
git clone -q --recursive https://github.com/microsoft/TRELLIS.2.git
git clone -q -b v0.4.0 https://github.com/NVlabs/nvdiffrast.git
git clone -q --recursive https://github.com/JeffreyXiang/CuMesh.git
git clone -q --recursive https://github.com/JeffreyXiang/FlexGEMM.git
git -C TRELLIS.2 rev-parse HEAD > $W/trellis2.commit
for d in nvdiffrast CuMesh FlexGEMM TRELLIS.2/o-voxel; do
  pip wheel -v --no-build-isolation --no-deps -w $W ./$d
done
ls -la $W
python - <<'P'
import os
from huggingface_hub import HfApi
HfApi().upload_folder(folder_path="/tmp/wheels", path_in_repo="wheels", repo_id=os.environ["IO_REPO"], repo_type="dataset")
P
