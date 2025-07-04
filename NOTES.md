# Notes

Docs:
- https://huggingface.co/docs/lerobot/getting_started_real_world_robot
- https://huggingface.co/docs/lerobot/so101

## Hardware info

`l1`: `/dev/tty.usbmodem5A4B0468311`
`f1`: `/dev/tty.usbmodem5A4B0468251`

## Environment setup

```sh
cd src
conda activate lerobot
export HF_LEROBOT_HOME=../hf-home
export HF_USER=$(huggingface-cli whoami | head -n 1)

export L1_PORT=/dev/tty.usbmodem5A4B0468311
export F1_PORT=/dev/tty.usbmodem5A4B0468251
```

### Camera configs

```sh
export CAMERA_CONFIG="{ base: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, top: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}, endeffector: {type: opencv, index_or_path: 2, width: 480, height: 640, fps: 30, rotation: -90} }"
```

## Calibration

```sh
# l1
python -m lerobot.calibrate \
    --teleop.type=so101_leader \
    --teleop.port=$L1_PORT \
    --teleop.id=l1

# f1
python -m lerobot.calibrate \
    --robot.type=so101_follower \
    --robot.port=$F1_PORT \
    --robot.id=f1
```

## Teleop

```sh
python -m lerobot.teleoperate \
    --robot.type=so101_follower \
    --robot.port=$F1_PORT \
    --robot.id=f1 \
    --teleop.type=so101_leader \
    --teleop.port=$L1_PORT \
    --teleop.id=l1
```

With cameras:

```sh
python -m lerobot.teleoperate \
    --robot.type=so101_follower \
    --robot.port=$F1_PORT \
    --robot.id=f1 \
    --robot.cameras="${CAMERA_CONFIG}" \
    --teleop.type=so101_leader \
    --teleop.port=$L1_PORT \
    --teleop.id=l1 \
    --display_data=true
```

## Record a dataset

```sh
HF_USER=$(huggingface-cli whoami | head -n 1)
echo "Hugging Face user: $HF_USER"
python -m lerobot.record \
    --robot.type=so101_follower \
    --robot.port=$F1_PORT \
    --robot.id=f1 \
    --robot.cameras="${CAMERA_CONFIG}" \
    --teleop.type=so101_leader \
    --teleop.port=$L1_PORT \
    --teleop.id=l1 \
    --display_data=true \
    --dataset.episode_time_s=120 \
    --dataset.reset_time_s=1 \
    --dataset.num_episodes=50 \
    --dataset.repo_id=${HF_USER}/so101_eraser_mat1 \
    --dataset.single_task="Grasp the eraser and move it to the mat."
```

Use `--resume=true` to resume the recording from the last episode.

### Recorded datasets

- [so101_box_pencil6](https://huggingface.co/un1c0rnio/so101_box_pencil6): Base, top, and end effector cameras
- [so101_eraser_mat1](https://huggingface.co/un1c0rnio/so101_eraser_mat1): "Grasp the eraser and move it to the mat."


## Teleop with [telegrip](https://github.com/DipFlip/telegrip)

Installation:

```sh
git clone https://github.com/DipFlip/telegrip.git
cd telegrip
conda install -c conda-forge pybullet
pip install -e .
```

Usage:

```sh
telegrip --left-port /dev/tty.usbmodem5A4B0468251 --right-port /dev/tty.usbmodem5A4B0468251 --log-level debug
```

Result: not working. The command queue doesn't appear to be processed.

## Training

One-time setup:

```sh
mkdir -p docker-home

# Add --user $(id -u):$(id -g) if using rootful docker
docker run --rm -it --gpus all -v $(pwd):/lerobot -v $(pwd)/docker-home:/docker-home -e HOME=/docker-home --shm-size=8g --workdir=/lerobot/src ghcr.io/ben-z/lerobot/gpu:main bash

# In the container
# Log into Hugging Face
huggingface-cli login
# Log into Weights & Biases
wandb login
```

Training:

```sh
CLUSTER_NAME=watgpu
HF_USER=$(huggingface-cli whoami | head -n 1)
EXP_SUFFIX="_${CLUSTER_NAME}_b64"

python -m lerobot.scripts.train \
  --dataset.repo_id=${HF_USER}/so101_eraser_mat1 \
  --policy.type=act \
  --policy.repo_id=${HF_USER}/act_so101_eraser_mat1 \
  --output_dir=../outputs/train/act_so101_eraser_mat1${EXP_SUFFIX} \
  --job_name=act_so101_eraser_mat1${EXP_SUFFIX} \
  --policy.device=cuda \
  --wandb.enable=true \
  --num_workers=4 \
  --batch_size=64 \
  --steps=800_000
```

```sh
# wato
docker run --rm -it --gpus all -v $(pwd):/lerobot -v $(pwd)/docker-home:/docker-home -e HOME=/docker-home -e CLUSTER_NAME=wato --shm-size=8g --workdir=/lerobot/src ghcr.io/ben-z/lerobot/gpu:main bash
# paper: Add --user $(id -u):$(id -g) for rootful docker, choose GPU with `--gpus "device=x"`
docker run --user $(id -u):$(id -g) --rm -it --gpus "device=3" -v $(pwd):/lerobot -v $(pwd)/docker-home:/docker-home -e HOME=/docker-home -e CLUSTER_NAME=paper --shm-size=8g --workdir=/lerobot/src ghcr.io/ben-z/lerobot/gpu:main bash

# In the container
# Adjust batch_size based on available VRAM
HF_USER=$(huggingface-cli whoami | head -n 1)
EXP_SUFFIX="_p40_b16"
echo "Hugging Face user: $HF_USER"
python -m lerobot.scripts.train \
  --dataset.repo_id=${HF_USER}/so101_eraser_mat1 \
  --policy.type=act \
  --policy.repo_id=${HF_USER}/act_so101_eraser_mat1 \
  --output_dir=../outputs/train/act_so101_eraser_mat1_${CLUSTER_NAME}${EXP_SUFFIX} \
  --job_name=act_so101_eraser_mat1_${CLUSTER_NAME}${EXP_SUFFIX} \
  --policy.device=cuda \
  --wandb.enable=true \
  --num_workers=4 \
  --batch_size=16 \
  --steps=800_000
```


## Evaluation

```sh
HF_USER=$(huggingface-cli whoami | head -n 1)
python -m lerobot.record  \
  --robot.type=so101_follower \
  --robot.port=$F1_PORT \
  --robot.cameras="${CAMERA_CONFIG}" \
  --robot.id=f1 \
  --teleop.type=so101_leader \
  --teleop.port=$L1_PORT \
  --teleop.id=l1 \
  --dataset.episode_time_s=120 \
  --dataset.reset_time_s=1 \
  --dataset.num_episodes=25 \
  --display_data=true \
  --dataset.repo_id=$HF_USER/eval_so101_eraser_mat1_040000 \
  --dataset.single_task="Grasp the eraser and move it to the mat." \
  --policy.path=${HF_USER}/act_so101_eraser_mat1_040000
```

As before, use `--resume=true` to resume the evaluation from the last episode.


## TODO

- [x] Figure out why camera resolution is not being set properly.
  - https://github.com/huggingface/lerobot/pull/1225






<!--
---------------------------------------------------------
MARK: Legacy notes
---------------------------------------------------------
-->

# Legacy Notes

## Setting up the leader motors:

```sh
python lerobot/scripts/configure_motor.py \
  --port /dev/tty.usbmodem5A4B0468311 \
  --brand feetech \
  --model sts3215 \
  --baudrate 1000000 \
  --ID 1

# repeat for other motors (2, 3, 4, 5, 6)
```


<details>

<summary>Logs</summary>

```
python lerobot/scripts/configure_motor.py \
  --port /dev/tty.usbmodem5A4B0468311 \
  --brand feetech \
  --model sts3215 \
  --baudrate 1000000 \
  --ID 1
Connected on port /dev/tty.usbmodem5A4B0468311
Scanning all baudrates and motor indices
100%|███████████████████████████████████████████████████████████████████████████████████████████| 9/9 [00:00<00:00, 16.43it/s]
Motor index found at: 1
Setting its index to desired index 1
Present Position [2050]
Offset [0]
Disconnected from motor bus.
~/Pr/lerobot main ⇡1 ❯ python lerobot/scripts/configure_motor.py \
  --port /dev/tty.usbmodem5A4B0468311 \
  --brand feetech \
  --model sts3215 \
  --baudrate 1000000 \
  --ID 2
Connected on port /dev/tty.usbmodem5A4B0468311
Scanning all baudrates and motor indices
100%|███████████████████████████████████████████████████████████████████| 9/9 [00:00<00:00, 16.44it/s]
Motor index found at: 1
Setting its index to desired index 2
Present Position [2049]
Offset [0]
Disconnected from motor bus.
~/Projects/lerobot main ⇡1 ?1 ❯ python lerobot/scripts/configure_motor.py --port /dev/tty.usbmodem5A4B0468311 --brand feetech --model sts3215 --baudrate 1000000 --ID 3
Connected on port /dev/tty.usbmodem5A4B0468311
Scanning all baudrates and motor indices
100%|███████████████████████████████████████████████████████████████████| 9/9 [00:00<00:00, 16.41it/s]
Motor index found at: 1
Setting its index to desired index 3
Present Position [2046]
Offset [0]
Disconnected from motor bus.
~/Projects/lerobot main ⇡1 ?1 ❯ python lerobot/scripts/configure_motor.py --port /dev/tty.usbmodem5A4B0468311 --brand feetech --model sts3215 --baudrate 1000000 --ID 4
Connected on port /dev/tty.usbmodem5A4B0468311
Scanning all baudrates and motor indices
100%|███████████████████████████████████████████████████████████████████| 9/9 [00:00<00:00, 16.44it/s]
Motor index found at: 1
Setting its index to desired index 4
Present Position [2050]
Offset [0]
Disconnected from motor bus.
~/Projects/lerobot main ⇡1 ?1 ❯ python lerobot/scripts/configure_motor.py --port /dev/tty.usbmodem5A4B0468311 --brand feetech --model sts3215 --baudrate 1000000 --ID 5
Connected on port /dev/tty.usbmodem5A4B0468311
Scanning all baudrates and motor indices
100%|███████████████████████████████████████████████████████████████████| 9/9 [00:00<00:00, 16.44it/s]
Motor index found at: 1
Setting its index to desired index 5
Present Position [2050]
Offset [0]
Disconnected from motor bus.
~/Projects/lerobot main ⇡1 ?1 ❯ python lerobot/scripts/configure_motor.py --port /dev/tty.usbmodem5A4B0468311 --brand feetech --model sts3215 --baudrate 1000000 --ID 6
Connected on port /dev/tty.usbmodem5A4B0468311
Scanning all baudrates and motor indices
100%|███████████████████████████████████████████████████████████████████| 9/9 [00:00<00:00, 16.44it/s]
Motor index found at: 1
Setting its index to desired index 6
Present Position [2051]
Offset [0]
Disconnected from motor bus.
```

</details>

## Teleop with cameras

```sh
python lerobot/scripts/control_robot.py \
  --robot.type=so101 \
  --control.type=teleoperate \
  --control.display_data=true
```

## Dataset recording

```sh
HF_USER=$(huggingface-cli whoami | head -n 1)
echo "Hugging Face user: $HF_USER"
python lerobot/scripts/control_robot.py \
  --robot.type=so101 \
  --control.type=record \
  --control.fps=30 \
  --control.single_task="Grasp a box and move it to the right side of the pencil." \
  --control.repo_id=${HF_USER}/so101_box_pencil5 \
  --control.tags='["so101"]' \
  --control.warmup_time_s=2 \
  --control.episode_time_s=30 \
  --control.reset_time_s=1 \
  --control.num_episodes=50 \
  --control.display_data=false \
  --control.push_to_hub=true
```

Optionally add `--control.resume=true` to resume the recording from the last episode.

### Recorded datasets

- [so101_box_pencil3](https://huggingface.co/un1c0rnio/so101_box_pencil3): Old dataset with old position on table and 3D printed table clamps, 2 cameras
- [so101_box_pencil4](https://huggingface.co/un1c0rnio/so101_box_pencil4): Better table clamps, new position on table, 2 cameras
- [so101_box_pencil5](https://huggingface.co/un1c0rnio/so101_box_pencil5): Additional end effector camera

## Training

On a Linux machine with an NVIDIA GPU, you can use the following command to run the training:

```sh
# Log into Hugging Face
huggingface-cli login

# Log into Weights & Biases
wandb login


HF_USER=$(huggingface-cli whoami | head -n 1)
echo "Hugging Face user: $HF_USER"

docker run --rm -it --gpus all -v $(pwd):/lerobot -v ~/.cache/huggingface:/root/.cache/huggingface -v ~/.config/wandb:/root/.config/wandb -v ~/.netrc:/root/.netrc --shm-size=8g --workdir=/lerobot/src ghcr.io/ben-z/lerobot/gpu:main python -m lerobot.scripts.train \
  --dataset.repo_id=${HF_USER}/so101_box_pencil5 \
  --policy.type=act \
  --output_dir=outputs/train/act_so101_box_pencil5 \
  --job_name=act_so101_box_pencil5_wato \
  --policy.device=cuda \
  --wandb.enable=true \
  --num_workers=8 \
  --batch_size=32
```

Or using SLURM:

```sh
conda activate lerobot
HF_USER=$(huggingface-cli whoami | head -n 1)
echo "Hugging Face user: $HF_USER"
sbatch --partition=compute_dense --cpus-per-task 8 --mem 14G --gres gpu:rtx_4090:1,tmpdisk:20480 --time 5-00:00:00 --wrap "slurm-start-dockerd.sh && DOCKER_HOST=unix:///tmp/run/docker.sock docker run --rm --gpus all -v $(pwd):/lerobot -v ~/.cache/huggingface:/root/.cache/huggingface -v ~/.config/wandb:/root/.config/wandb -v ~/.netrc:/root/.netrc --shm-size=8g --workdir=/lerobot/src ghcr.io/ben-z/lerobot/gpu:main python -m lerobot.scripts.train \
  --dataset.repo_id=${HF_USER}/so101_eraser_mat1 \
  --policy.type=act \
  --policy.repo_id=${HF_USER}/act_so101_eraser_mat1 \
  --output_dir=outputs/train/act_so101_eraser_mat1 \
  --job_name=act_so101_eraser_mat1_wato \
  --policy.device=cuda \
  --wandb.enable=true \
  --num_workers=4 \
  --batch_size=32 \
  --steps=400_000"
```

To resume training from a checkpoint, use `--resume=true`:

```sh
python -m lerobot.scripts.train \
  --config_path=outputs/train/act_so101_eraser_mat1/checkpoints/last/pretrained_model/train_config.json \
  --resume=true

# or in SLURM
sbatch --partition=compute_dense --cpus-per-task 8 --mem 14G --gres gpu:rtx_4090:1,tmpdisk:20480 --time 5-00:00:00 --wrap "slurm-start-dockerd.sh && DOCKER_HOST=unix:///tmp/run/docker.sock docker run --rm --gpus all -v $(pwd):/lerobot -v ~/.cache/huggingface:/root/.cache/huggingface -v ~/.config/wandb:/root/.config/wandb -v ~/.netrc:/root/.netrc --shm-size=8g --workdir=/lerobot/src ghcr.io/ben-z/lerobot/gpu:main python -m lerobot.scripts.train \
  --config_path=outputs/train/act_so101_eraser_mat1/checkpoints/last/pretrained_model/train_config.json \
  --resume=true"
```

Upload the trained model to the Hugging Face hub:

```sh
huggingface-cli upload ${HF_USER}/act_so101_eraser_mat1 \
  outputs/train/act_so101_eraser_mat1/checkpoints/last/pretrained_model
```

Upload a checkpoint only:

```sh
# list checkpoints
ls outputs/train/act_so101_eraser_mat1/checkpoints

HF_USER=$(huggingface-cli whoami | head -n 1)
CKPT=040000
huggingface-cli upload ${HF_USER}/act_so101_eraser_mat1_${CKPT} \
  outputs/train/act_so101_eraser_mat1/checkpoints/${CKPT}/pretrained_model
```

Training [SmolVLA](https://huggingface.co/blog/smolvla):

```sh
conda activate lerobot
HF_USER=$(huggingface-cli whoami | head -n 1)
echo "Hugging Face user: $HF_USER"
sbatch --partition=compute_dense --cpus-per-task 8 --mem 14G --gres gpu:rtx_4090:1,tmpdisk:20480 --time 5-00:00:00 --wrap "slurm-start-dockerd.sh && DOCKER_HOST=unix:///tmp/run/docker.sock docker run --rm --gpus all -v $(pwd):/lerobot -v ~/.cache/huggingface:/root/.cache/huggingface -v ~/.config/wandb:/root/.config/wandb -v ~/.netrc:/root/.netrc --shm-size=8g --workdir /lerobot/src ghcr.io/ben-z/lerobot/gpu-dev2:main python -m lerobot.scripts.train \
  --dataset.repo_id=${HF_USER}/so101_eraser_mat1 \
  --policy.path=lerobot/smolvla_base \
  --policy.repo_id=${HF_USER}/smolvla_so101_eraser_mat1 \
  --output_dir=outputs/train/smolvla_so101_eraser_mat1 \
  --job_name=smolvla_so101_eraser_mat1_wato \
  --policy.device=cuda \
  --wandb.enable=true \
  --num_workers=2 \
  --batch_size=64 \
  --steps=200_000"
```

## Evaluation

### With a local model

```sh
HF_USER=$(huggingface-cli whoami | head -n 1)
echo "Hugging Face user: $HF_USER"
python lerobot/scripts/control_robot.py \
  --robot.type=so101 \
  --control.type=record \
  --control.fps=30 \
  --control.single_task="Grasp a box and move it to the right side of the pencil." \
  --control.repo_id=${HF_USER}/eval_act_so101_test \
  --control.tags='["so101","tutorial"]' \
  --control.warmup_time_s=2 \
  --control.episode_time_s=30 \
  --control.reset_time_s=2 \
  --control.num_episodes=1 \
  --control.push_to_hub=false \
  --control.policy.path=outputs/train/act_so101_test/checkpoints/last/pretrained_model
```

### With a model from the Hugging Face hub


```sh
HF_USER=$(huggingface-cli whoami | head -n 1)
echo "Hugging Face user: $HF_USER"
python lerobot/scripts/control_robot.py \
  --robot.type=so101 \
  --control.type=record \
  --control.fps=30 \
  --control.single_task="Grasp a box and move it to the right side of the pencil." \
  --control.repo_id=${HF_USER}/eval_act_so101_box_pencil5_040000 \
  --control.tags='["so101"]' \
  --control.warmup_time_s=2 \
  --control.episode_time_s=60 \
  --control.reset_time_s=2 \
  --control.num_episodes=10 \
  --control.push_to_hub=true \
  --control.policy.path=${HF_USER}/act_so101_box_pencil5_040000 \
  --control.display_data=true \
  --control.policy.device=mps
```

Add `--control.resume=true` to resume in the same repo.

### Notes

- The smolvla policy appears to run at 2fps instead of 30fps due to compute constraints. And the lack of damping appears to halt progress. When the arm is manually held down (add damping to reduce shaking), it slowly executes the task.

## Visualize a dataset

```sh
python lerobot/scripts/visualize_dataset_html.py --repo-id un1c0rnio/eval_act_so101_box_pencil5
```

Visualize online dataset: https://huggingface.co/spaces/lerobot/visualize_dataset
