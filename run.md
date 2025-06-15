### Resources
```shell 
https://github.com/ben-z/lerobot
https://huggingface.co/datasets/littledragon/so101_sock_stowing2
https://huggingface.co/datasets/littledragon/so101_sock_stowing3
```


### Env 
```shell
# Log into Hugging Face
huggingface-cli login

# Log into Weights & Biases
wandb login

HF_USER=$(huggingface-cli whoami | head -n 1)
echo "Hugging Face user: $HF_USER"

HF_USER='littledragon'
export PATH=/root/bin:/lustre/fsw/portfolios/adlr/projects/adlr_other_infra/release/cluster-interface/latest:/root/bin:/lustre/fsw/portfolios/adlr/projects/adlr_other_infra/release/cluster-interface/latest:/root/bin:/opt/amazon/openmpi/bin:/opt/amazon/efa/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/root/miniconda3/condabin

export LD_LIBRARY_PATH="/root/miniconda3/lib:$LD_LIBRARY_PATH" && echo $LD_LIBRARY_PATH
```
### Download dataset
```shell
huggingface-cli download \
    --repo-type dataset littledragon/so101_sock_stowing2 \
    --local-dir ./demo_data/littledragon/so101_sock_stowing2

huggingface-cli download \
    --repo-type dataset littledragon/so101_sock_stowing3 \
    --local-dir ./demo_data/littledragon/so101_sock_stowing3

huggingface-cli download \
    --repo-type dataset un1c0rnio/so101_sock_stowing_3pair2 \
    --local-dir ./demo_data/un1c0rnio/so101_sock_stowing_3pair2
```

### Data conversion
```shell
python draw_box/video_postprocess.py 

python draw_box/video_postprocess.py --video_root=/Users/xiaoli/projects/code/lerobot/demo_data/un1c0rnio/so101_sock_stowing_3pair2/videos/chunk-000/observation.images.top

python draw_box/video_postprocess.py --video_root=/Users/xiaoli/projects/code/lerobot/demo_data/littledragon/so101_sock_stowing2/videos/chunk-000/observation.images.top
```

### Upload data
```shell
huggingface-cli upload littledragon/so101_sock_stowing3 ./hf-home/littledragon/so101_sock_stowing3 . --repo-type dataset

huggingface-cli upload littledragon/so101_sock_stowing2_boxes ./demo_data/littledragon/so101_sock_stowing2 . --repo-type dataset

huggingface-cli upload littledragon/so101_sock_stowing3_boxes ./demo_data/littledragon/so101_sock_stowing3 . --repo-type dataset

huggingface-cli upload littledragon/so101_sock_stowing_3pair2_boxes ./demo_data/un1c0rnio/so101_sock_stowing_3pair2 . --repo-type dataset

```

### Training
```shell

HF_USER='littledragon'
python lerobot/scripts/train.py \
  --dataset.repo_id=${HF_USER}/so101_sock_stowing2 \
  --policy.type=act \
  --output_dir=outputs/train/act_so101_sock_stowing_ft \
  --job_name=sock_stowing_ft \
  --policy.device=cuda \
  --wandb.enable=true


# with smolVLA
HF_USER='littledragon'
python lerobot/scripts/train.py \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=${HF_USER}/so101_sock_stowing2\
  --batch_size=2 \
  --steps=20000 \
  --output_dir=outputs/train/smolvla_so101_sock_stowing_ft0 \
  --job_name=my_smolvla_training \
  --policy.device=cuda \
  --wandb.enable=true
```

### Upload
```shell
HF_USER='littledragon'
huggingface-cli upload ${HF_USER}/so101_act_sock_stowing2 \
  outputs/train/act_so101_sock_stowing_ft/checkpoints/last/pretrained_model
```