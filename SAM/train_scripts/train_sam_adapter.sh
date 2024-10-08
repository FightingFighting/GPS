#!/bin/bash


#Set job requirements
#SBATCH -n 1
#SBATCH -t 24:00:00
#SBATCH -p gpu_a100
#SBATCH --gpus-per-node=1
#SBATCH --job-name=sam_adapter
#SBATCH --output=slurm_output_%A_sam_adapter.out



source activate sam

python -m torch.distributed.launch \
--nnodes 1 --nproc_per_node 1 \
--master_port=29512 \
train.py --config ./configs/cod-sam-vit-b.yaml \
--name polyp_adapter --tuning-mode adapter --gpu_id 0 --optimizer adamw