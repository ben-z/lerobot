# Notes

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

## Training

On a Linux machine with an NVIDIA GPU, you can use the following command to run the training:

```sh
# Log into Hugging Face
huggingface-cli login

# Log into Weights & Biases
wandb login

docker run --rm -it --gpus all -v $(pwd):/lerobot -v ~/.cache/huggingface:/root/.cache/huggingface -v ~/.config/wandb:/root/.config/wandb -v ~/.netrc:/root/.netrc --shm-size=4g ghcr.io/ben-z/lerobot/gpu:main bash

HF_USER=$(huggingface-cli whoami | head -n 1)
echo "Hugging Face user: $HF_USER"
python lerobot/scripts/train.py \
  --dataset.repo_id=${HF_USER}/so101_test \
  --policy.type=act \
  --output_dir=outputs/train/act_so101_test \
  --job_name=act_so101_test_wato \
  --policy.device=cuda \
  --wandb.enable=true
```
