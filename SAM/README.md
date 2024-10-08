## Polyp Segmentation (kvasir-SEG) on SAM using GPS 

## Environment
The code is based on [SAM Adapter](https://github.com/tianrun-chen/SAM-Adapter-PyTorch). Please follow them for the installation.

## Dataset
Download kvasir-SEG dataset  from [Kvasir](https://datasets.simula.no/kvasir-seg/), and unzip and put it in `data`.


# Train
For our GPS method, you can run:
```
bash train_scripts/train_sam_gps.sh
```
For other methods, like full, adapter, and ssf, you can also run the corresponding scripts in the `train_scripts`.


## evaluation
```
bash test_scripts/test_sam_gps.sh
```

## Acknowledgements
The part of the code is derived from [SAM Adapter](https://github.com/tianrun-chen/SAM-Adapter-PyTorch). Our experiment setting also follow them.
