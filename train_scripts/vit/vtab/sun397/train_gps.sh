gpu_id=4
bz=16
lr=0.0015

python train_gps.py /online1/ycsc_pku/liujm/zqz/data/VTAB/sun397 \
    --dataset sun397 \
    --num-classes 397 --direct-resize --no-aug \
    --model vit_base_patch16_224_in21k \
    --epochs 100 \
    --batch-size $bz \
    --opt adam  --weight-decay 0 \
    --warmup-lr 1e-7 --warmup-epochs 10 \
    --lr $lr --min-lr 1e-8 \
    --drop-path 0 --img-size 224 \
	--mixup 0 --cutmix 0 --smoothing 0 \
    --output output/ \
    --amp --tuning-mode part --pretrained \
    --pruning --pruning_method gradient_perCell \
    --times_para 1 \
    --gpu_id $gpu_id \
    --log-wandb \
    --experiment vtab \
    --contrast-aug --no-prefetcher --contrastive
