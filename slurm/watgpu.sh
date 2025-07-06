#!/bin/bash

#SBATCH --job-name=robo
# watgpu208: RTX 6000 Ada (48GB)
# watgpu508: H200 (141GB)
#SBATCH --nodelist=watgpu208
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

# act
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
  --optimizer.lr=${LR} \
  --wandb.enable=true \
  --num_workers=8 \
  --batch_size=${BATCH_SIZE} \
  --steps=800_000 \
  --save_freq=5_000

# # smolvla
# EXP_SUFFIX="_${CLUSTER_NAME}_h200_b256"
# python -m lerobot.scripts.train \
#   --dataset.repo_id=${HF_USER}/so101_eraser_mat1 \
#   --policy.path=lerobot/smolvla_base \
#   --policy.repo_id=${HF_USER}/smolvla_so101_eraser_mat1 \
#   --output_dir=../outputs/train/smolvla_so101_eraser_mat1${EXP_SUFFIX} \
#   --job_name=smolvla_so101_eraser_mat1${EXP_SUFFIX} \
#   --policy.device=cuda \
#   --wandb.enable=true \
#   --num_workers=8 \
#   --batch_size=256 \
#   --steps=800_000 \
#   --save_freq=5_000