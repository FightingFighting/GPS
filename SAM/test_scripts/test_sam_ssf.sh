#!/bin/bash


#Set job requirements
#SBATCH -n 1
#SBATCH -t 24:00:00
#SBATCH -p gpu_a100
#SBATCH --gpus-per-node=1
#SBATCH --job-name=sam_ssf
#SBATCH --output=slurm_output_%A_sam_ssf.out



source activate sam




python test.py --config ./configs/cod-sam-vit-b.yaml \
--model ./save/polyp_ssf/model_epoch_best.pth
