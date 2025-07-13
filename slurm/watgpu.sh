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
##SBATCH --nodelist=watgpu502

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=256G
#SBATCH --time=3-00:00

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

DATASET_NAME="so101_die_mat3"
DATASET_REPO_ID="${HF_USER}/${DATASET_NAME}"

# # ACT
# # Tuning tips: https://github.com/tonyzhaozh/act/blob/742c753c0d4a5d87076c8f69e5628c79a8cc5488/README.md#new-act-tuning-tips
# # 64 uses ~47GiB VRAM, 128 uses ~93GiB VRAM
# BATCH_SIZE=64
# LR=5e-5
# POLICY_REPO_ID="${HF_USER}/act_${DATASET_NAME}_b${BATCH_SIZE}_lr${LR}_${SLURM_JOB_NAME}"
# WANDB_NOTES="batch_size=${BATCH_SIZE}, lr=${LR}"
# OUTPUT_DIR="../outputs/train/${POLICY_REPO_ID}"
# try_resume "${OUTPUT_DIR}"
# python -m lerobot.scripts.train \
#   --dataset.repo_id="${DATASET_REPO_ID}" \
#   --policy.type=act \
#   --policy.repo_id="${POLICY_REPO_ID}" \
#   --output_dir="${OUTPUT_DIR}" \
#   --job_name="${POLICY_REPO_ID}_${CLUSTER_NAME}" \
#   --policy.device=cuda \
#   --policy.optimizer_lr="${LR}" \
#   --wandb.enable=true \
#   --wandb.notes="${WANDB_NOTES}" \
#   --num_workers=8 \
#   --batch_size="${BATCH_SIZE}" \
#   --steps="800_000" \
#   --save_freq="5_000"

# SmolVLA
# 128 uses ~43GiB VRAM
BATCH_SIZE=128
LR=1e-3
POLICY_REPO_ID="${HF_USER}/smolvla_${DATASET_NAME}_b${BATCH_SIZE}_lr${LR}_${SLURM_JOB_NAME}"
OUTPUT_DIR="../outputs/train/${POLICY_REPO_ID}"
try_resume "${OUTPUT_DIR}"
python -m lerobot.scripts.train \
  --dataset.repo_id="${DATASET_REPO_ID}" \
  --policy.path=lerobot/smolvla_base \
  --policy.repo_id="${POLICY_REPO_ID}" \
  --output_dir="${OUTPUT_DIR}" \
  --job_name="${POLICY_REPO_ID}_${CLUSTER_NAME}" \
  --policy.device=cuda \
  --policy.optimizer_lr="${LR}" \
  --wandb.enable=true \
  --num_workers=8 \
  --batch_size="${BATCH_SIZE}" \
  --steps="800_000" \
  --save_freq="5_000"
