#!/bin/bash

#SBATCH --job-name=robo

# watgpu102: RTX 6000 Ada x2 (x9 in lspci) (49140MiB ~48GiB)
# watgpu108: RTX 6000 Ada x8 (49140MiB ~48GiB)
# watgpu208: RTX 6000 Ada x8 (49140MiB ~48GiB)
# watgpu308: L40S x4 (46068MiB ~45GiB), RTX A6000 x2 (49140MiB ~48GiB), RTX 6000 Ada x2 (49140MiB ~48GiB)
# watgpu408: L40S x8 (46068MiB ~45GiB)
# watgpu502: H200 x2? (x6 in lspci) (143771MiB ~140GiB)
# watgpu508: H200 x4? (x6 in lspci) (143771MiB ~140GiB)
# watgpu608: RTX 6000 Ada x4 (49140MiB ~48GiB), L40S x2 (46068MiB ~45GiB)
##SBATCH --nodelist=watgpu308

# Exclude L40S nodes because their VRAM is too low for the job
##SBATCH --exclude=watgpu308,watgpu408,watgpu608

# Exclude L40S/A6000/6000Ada nodes because their VRAM is too low for the job
#SBATCH --exclude=watgpu102,watgpu108,watgpu208,watgpu308,watgpu408,watgpu608

# Exclude GPUs with uncorrectable ECC errors
##SBATCH --exclude=watgpu408

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=7-00:00

source /opt/anaconda3/etc/profile.d/conda.sh
conda activate lerobot

function try_resume() {
  local __output_dir=${1:?}

  echo "INFO: SLURM_RESTART_COUNT=${SLURM_RESTART_COUNT}"

  if [[ "${SLURM_RESTART_COUNT}" -ge 1 ]]; then
    echo "INFO: This job was restarted. Checking for checkpoints..."

    if [ -d "${__output_dir}" ]; then
      echo "INFO: Found output directory: ${__output_dir}."

      __ckpt_json="${__output_dir}/checkpoints/last/pretrained_model/train_config.json"
      if [ -f "${__ckpt_json}" ]; then
        echo "INFO: Found checkpoint config: ${__ckpt_json}. Resuming training..."

        python -m lerobot.scripts.train \
          --config_path="${__ckpt_json}" \
          --resume=true
        echo "INFO: Done!"
        exit 0
      else
        echo "WARNING: No checkpoint config found in ${__output_dir}. The job likely crashed before saving a checkpoint. Deleting the output directory and starting over."
        echo "INFO: Waiting for 60 seconds before deleting '${__output_dir}'. Cancel the job to abort."
        sleep 60
        echo "INFO: Deleting '${__output_dir}'."
        rm -rf "${__output_dir}"
      fi
    else
      echo "INFO: No output directory found in ${__output_dir}. Continuing as normal."
    fi
  else
    echo "INFO: This job was not restarted. Continuing as normal."
  fi
}

cd src

# MARK: new training
CLUSTER_NAME=watgpu

HF_USER=$(huggingface-cli whoami | head -n 1)
echo "Hugging Face user: $HF_USER"

DATASET_NAME="so101_cloth_folding1"
DATASET_REPO_ID="${HF_USER}/${DATASET_NAME}"

# # ACT
# POLICY_TYPE="act"
# # Tuning tips: https://github.com/tonyzhaozh/act/blob/742c753c0d4a5d87076c8f69e5628c79a8cc5488/README.md#new-act-tuning-tips
# # 64 uses ~47GiB VRAM, 128 uses ~93GiB VRAM
# BATCH_SIZE=64
# LR=1e-4
# POLICY_REPO_ID="${HF_USER}/act_${DATASET_NAME}_b${BATCH_SIZE}_lr${LR}_${SLURM_JOB_NAME}"
# WANDB_NOTES="batch_size=${BATCH_SIZE}, lr=${LR}"
# OUTPUT_DIR="../outputs/train/${POLICY_REPO_ID}"
# try_resume "${OUTPUT_DIR}"
# python -m lerobot.scripts.train \
#   --dataset.repo_id="${DATASET_REPO_ID}" \
#   --policy.type="${POLICY_TYPE}" \
#   --policy.repo_id="${POLICY_REPO_ID}" \
#   --output_dir="${OUTPUT_DIR}" \
#   --job_name="${POLICY_REPO_ID}_${CLUSTER_NAME}" \
#   --policy.device=cuda \
#   --policy.optimizer_lr="${LR}" \
#   --wandb.enable=true \
#   --wandb.project="lerobot-${POLICY_TYPE}" \
#   --wandb.notes="${WANDB_NOTES}" \
#   --wandb.disable_artifact=true \
#   --num_workers=8 \
#   --batch_size="${BATCH_SIZE}" \
#   --steps="800_000" \
#   --save_freq="10_000"

# # SmolVLA
# POLICY_TYPE="smolvla"
# # batch_size=128 uses ~43GiB VRAM with chunk_size=50
# # batch_size=64 uses ~25GiB VRAM with chunk_size=100
# BATCH_SIZE=64
# LR=5e-4
# CHUNK_SIZE=100
# N_ACTION_STEPS=100
# POLICY_REPO_ID="${HF_USER}/smolvla_${DATASET_NAME}_b${BATCH_SIZE}_lr${LR}_cs${CHUNK_SIZE}_nas${N_ACTION_STEPS}_${SLURM_JOB_NAME}"
# WANDB_NOTES="batch_size=${BATCH_SIZE}, lr=${LR}, chunk_size=${CHUNK_SIZE}, n_action_steps=${N_ACTION_STEPS}"
# OUTPUT_DIR="../outputs/train/${POLICY_REPO_ID}"
# try_resume "${OUTPUT_DIR}"
# python -m lerobot.scripts.train \
#   --dataset.repo_id="${DATASET_REPO_ID}" \
#   --policy.path=lerobot/smolvla_base \
#   --policy.repo_id="${POLICY_REPO_ID}" \
#   --output_dir="${OUTPUT_DIR}" \
#   --job_name="${POLICY_REPO_ID}_${CLUSTER_NAME}" \
#   --policy.device=cuda \
#   --policy.optimizer_lr="${LR}" \
#   --policy.chunk_size="${CHUNK_SIZE}" \
#   --policy.n_action_steps="${N_ACTION_STEPS}" \
#   --wandb.enable=true \
#   --wandb.project="lerobot-${POLICY_TYPE}" \
#   --wandb.disable_artifact=true \
#   --wandb.notes="${WANDB_NOTES}" \
#   --num_workers=8 \
#   --batch_size="${BATCH_SIZE}" \
#   --steps="800_000" \
#   --save_freq="10_000"

# pi0fast
POLICY_TYPE="pi0fast"
# Default batch_size=8, chunk_size=10 uses ~44-47GiB VRAM (doesn't fit in L40S. Fits in RTX 6000 Ada for a few steps (~3K), but OOMs later.)
# batch_size=4 uses ~35-40GiB VRAM. It seems to be different each run.
# batch_size=8, chunk_size=50, n_action_steps=50 uses ~54GiB VRAM
# batch_size=24, chunk_size=100, n_action_steps=100 uses ~137GiB VRAM
BATCH_SIZE=24
LR=1e-4
CHUNK_SIZE=100
N_ACTION_STEPS=100
POLICY_REPO_ID="${HF_USER}/pi0fast_${DATASET_NAME}_b${BATCH_SIZE}_lr${LR}_cs${CHUNK_SIZE}_nas${N_ACTION_STEPS}_${SLURM_JOB_NAME}"
WANDB_NOTES="batch_size=${BATCH_SIZE}, lr=${LR}, chunk_size=${CHUNK_SIZE}, n_action_steps=${N_ACTION_STEPS}"
OUTPUT_DIR="../outputs/train/${POLICY_REPO_ID}"
try_resume "${OUTPUT_DIR}"
python -m lerobot.scripts.train \
  --dataset.repo_id="${DATASET_REPO_ID}" \
  --policy.path=lerobot/pi0fast_base \
  --policy.repo_id="${POLICY_REPO_ID}" \
  --output_dir="${OUTPUT_DIR}" \
  --job_name="${POLICY_REPO_ID}_${CLUSTER_NAME}" \
  --policy.device=cuda \
  --policy.optimizer_lr="${LR}" \
  --policy.chunk_size="${CHUNK_SIZE}" \
  --policy.n_action_steps="${N_ACTION_STEPS}" \
  --wandb.enable=true \
  --wandb.project="lerobot-${POLICY_TYPE}" \
  --wandb.disable_artifact=true \
  --wandb.notes="${WANDB_NOTES}" \
  --num_workers=8 \
  --batch_size="${BATCH_SIZE}" \
  --steps="800_000" \
  --save_freq="10_000"
