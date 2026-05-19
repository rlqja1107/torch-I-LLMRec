#!/usr/bin/env bash
set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"

conda create -n i_llmrec python=3.10 -y
conda activate i_llmrec

pip install torch==2.0.1 torchvision==0.15.2 # CUDA: 11.7
pip install transformers==4.36.2 sentence_transformers==3.0.1 deepspeed==0.9.5 accelerate==0.27.2
pip install "setuptools<70" packaging ninja wheel
pip install flash-attn==2.4.2 --no-build-isolation # CUDA version should upper than 11.7
pip install git+https://github.com/bfshi/scaling_on_scales.git
pip install sentencepiece==0.1.99
pip install protobuf==3.20.*
pip install numpy==1.26.4
pip install tqdm gdown
