#!/bin/bash

#SBATCH --job-name=robo
#SBATCH --gres=gpu:mligpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=3-00:00

conda activate lerobot

cd src
CLUSTER_NAME=watgpu
EXP_SUFFIX="_${CLUSTER_NAME}_h200_b128"

HF_USER=$(huggingface-cli whoami | head -n 1)
echo "Hugging Face user: $HF_USER"

python -m lerobot.scripts.train \
  --dataset.repo_id=${HF_USER}/so101_eraser_mat1 \
  --policy.type=act \
  --policy.repo_id=${HF_USER}/act_so101_eraser_mat1 \
  --output_dir=../outputs/train/act_so101_eraser_mat1${EXP_SUFFIX} \
  --job_name=act_so101_eraser_mat1${EXP_SUFFIX} \
  --policy.device=cuda \
  --wandb.enable=true \
  --num_workers=4 \
  --batch_size=128 \
  --steps=100_000 \
  --save_freq=1_000
