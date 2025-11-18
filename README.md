We have a new finding for selection-based parameter efficient fine-tuning (PEFT). Please read our new paper: **[NeuroAda: Activating Each Neuron's Potential for Parameter-Efficient Fine-Tuning](https://arxiv.org/abs/2510.18940)** and see our official repository: **[NeuroAda](https://github.com/FightingFighting/NeuroAda)**


# GPS
This is an official repository for our CVPR 2024 paper: [Gradient-based Parameter Selection for Efficient Fine-Tuning](https://arxiv.org/pdf/2312.10136)

For the segmentation Task on the SAM model using our GPS method, please see [SAM GPS](https://github.com/FightingFighting/GPS/tree/main/SAM).

## Environment
Please follow [SSF](https://github.com/dongzelian/SSF) for installation.

## Datasets
### FGVC
Please follow [VPT](https://github.com/KMnP/vpt) to download them.

You can also download them from [baiduyun](https://pan.baidu.com/s/15KWOBREl4oH-yObile3iFQ?pwd=nc9f) code: nc9f or [google drive](https://drive.google.com/drive/folders/110TEuKscYJdsb3EbddzBsInotxyMnLKB?usp=sharing)

If you only want to download the annotation for split, please download it from [fgvc_split.zip](https://github.com/user-attachments/files/17444450/fgvc_split.zip)



### VTAB 
Please follow [SSF](https://github.com/dongzelian/SSF) to download them.

You can also download them from [baiduyun](https://pan.baidu.com/s/1rCMDz1bEXBA7O7dMclD29w?pwd=r1s7) code: r1s7, or [google drive](https://drive.google.com/file/d/14dTPCoWOQ_4vcPYFvjXpAvBcoqtInEZz/view?usp=sharing)

If you only want to download the annotation for split, please download from [vtab-1k_split.zip](https://github.com/user-attachments/files/17444442/vtab-1k_split.zip)


## Train
Take the Stanford Cars task in FGVC for example:
1. Replace `/path/to/FGVC/` with your path of the FGVC dataset in `train_scripts/vit/fgvc/stanford_cars.sh`
2. cd GPS
3. run
   `bash train_scripts/vit/fgvc/stanford_cars.sh`

For the VTAB task, please see the scripts in `train_scripts/vit/vtab`. We have already updated the scripts for VTAB.
   
## Cite
If this project is helpful for you, please cite our paper:
```
@inproceedings{zhang2024gradient,
  title={Gradient-based Parameter Selection for Efficient Fine-Tuning},
  author={Zhang, Zhi and Zhang, Qizhe and Gao, Zijun and Zhang, Renrui and Shutova, Ekaterina and Zhou, Shiji and Zhang, Shanghang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={28566--28577},
  year={2024}
}
```


## Acknowledgement
Our experiment follows [SSF](https://github.com/dongzelian/SSF).

The code is built upon [SSF](https://github.com/dongzelian/SSF) and [VPT](https://github.com/KMnP/vpt).


