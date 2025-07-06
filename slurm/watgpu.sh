#!/bin/bash

#SBATCH --job-name=robo
# watgpu108: RTX 6000 Ada x8 (49140MiB ~48GiB)
# watgpu208: RTX 6000 Ada x8 (49140MiB ~48GiB)
# watgpu308: L40S x4 (46068MiB ~45GiB), RTX A6000 x2 (49140MiB ~48GiB), RTX 6000 Ada x2 (49140MiB ~48GiB)
# watgpu408: L40S x8 (46068MiB ~45GiB)
# watgpu508: H200 x4? (143771MiB ~140GiB)
#SBATCH --nodelist=watgpu308
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=256G
#SBATCH --time=3-00:00

source /opt/anaconda3/etc/profile.d/conda.sh
conda activate lerobot

cd src
CLUSTER_NAME=watgpu

HF_USER=$(huggingface-cli whoami | head -n 1)
echo "Hugging Face user: $HF_USER"

DATASET_NAME="so101_die_mat1"
DATASET_REPO_ID=${HF_USER}/${DATASET_NAME}

# ACT
# Tuning tips: https://github.com/tonyzhaozh/act/blob/742c753c0d4a5d87076c8f69e5628c79a8cc5488/README.md#new-act-tuning-tips
# 64 uses ~47GiB VRAM, 128 uses ~93GiB VRAM
BATCH_SIZE=64
LR=5e-5
POLICY_REPO_ID="${HF_USER}/act_${DATASET_NAME}_b${BATCH_SIZE}"
python -m lerobot.scripts.train \
  --dataset.repo_id=${DATASET_REPO_ID} \
  --policy.type=act \
  --policy.repo_id=${POLICY_REPO_ID} \
  --output_dir=../outputs/train/${POLICY_REPO_ID} \
  --job_name=${POLICY_REPO_ID}_${CLUSTER_NAME} \
  --policy.device=cuda \
  --policy.chunk_size=30 \
  --policy.n_action_steps=30 \
  --optimizer.lr=${LR} \
  --wandb.enable=true \
  --num_workers=8 \
  --batch_size=${BATCH_SIZE} \
  --steps=800_000 \
  --save_freq=5_000

# # SmolVLA
# 128 uses ~43GiB VRAM
# BATCH_SIZE=128
# LR=5e-4
# POLICY_REPO_ID="${HF_USER}/smolvla_${DATASET_NAME}_b${BATCH_SIZE}"
# python -m lerobot.scripts.train \
#   --dataset.repo_id=${DATASET_REPO_ID} \
#   --policy.path=lerobot/smolvla_base \
#   --policy.repo_id=${POLICY_REPO_ID} \
#   --output_dir=../outputs/train/${POLICY_REPO_ID} \
#   --job_name=${POLICY_REPO_ID}_${CLUSTER_NAME} \
#   --policy.device=cuda \
#   --wandb.enable=true \
#   --num_workers=8 \
#   --batch_size=${BATCH_SIZE} \
#   --optimizer.lr=${LR} \
#   --steps=800_000 \
#   --save_freq=5_000
