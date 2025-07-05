#!/bin/bash

#SBATCH --job-name=robo
#SBATCH --nodelist=watgpu508
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=3-00:00

source /opt/anaconda3/etc/profile.d/conda.sh
conda activate lerobot

cd src
CLUSTER_NAME=watgpu

HF_USER=$(huggingface-cli whoami | head -n 1)
echo "Hugging Face user: $HF_USER"

# # act
# EXP_SUFFIX="_${CLUSTER_NAME}_h200_b128"
# python -m lerobot.scripts.train \
#   --dataset.repo_id=${HF_USER}/so101_eraser_mat1 \
#   --policy.type=act \
#   --policy.repo_id=${HF_USER}/act_so101_eraser_mat1 \
#   --output_dir=../outputs/train/act_so101_eraser_mat1${EXP_SUFFIX} \
#   --job_name=act_so101_eraser_mat1${EXP_SUFFIX} \
#   --policy.device=cuda \
#   --wandb.enable=true \
#   --num_workers=8 \
#   --batch_size=128 \
#   --steps=100_000 \
#   --save_freq=1_000

# smolvla
EXP_SUFFIX="_${CLUSTER_NAME}_h200_b256"
python -m lerobot.scripts.train \
  --dataset.repo_id=${HF_USER}/so101_eraser_mat1 \
  --policy.path=lerobot/smolvla_base \
  --output_dir=../outputs/train/smolvla_so101_eraser_mat1${EXP_SUFFIX} \
  --job_name=smolvla_so101_eraser_mat1${EXP_SUFFIX} \
  --policy.device=cuda \
  --wandb.enable=true \
  --num_workers=2 \
  --batch_size=256 \
  --steps=200_000 \
  --save_freq=1_000