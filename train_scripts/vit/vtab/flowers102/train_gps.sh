gpu_id=0
bz=16
lr=0.002

python train_gps.py path/to/vtab-1k/oxford_flowers102 \
    --dataset flowers102 \
    --num-classes 102 --direct-resize --no-aug \
    --model vit_base_patch16_224_in21k \
    --epochs 100 \
    --batch-size $bz \
    --opt adam  --weight-decay 0 \
    --warmup-lr 1e-7 --warmup-epochs 10 \
    --lr $lr --min-lr 1e-8 \
    --drop-path 0 --img-size 224 \
	  --mixup 0 --cutmix 0 --smoothing 0 \
    --output /gpfs/home6/zzhang3/gps_git/output/ \
    --amp --tuning-mode part --pretrained \
    --pruning --pruning_method gradient_perCell \
    --times_para 1 \
    --gpu_id $gpu_id \
    --log-wandb \
    --experiment vtab \
    --contrast-aug --no-prefetcher --contrastiveve
