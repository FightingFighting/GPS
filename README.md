# GPS
This is an official repository for the paper: [Gradient-based Parameter Selection for Efficient Fine-Tuning](https://arxiv.org/pdf/2312.10136)

# Environment
Please follow [SSF](https://github.com/dongzelian/SSF) for installation.

# Datasets
## FGVC
Please follow [VPT](https://github.com/KMnP/vpt) to download them.
## VTAB 
Please follow [SSF](https://github.com/dongzelian/SSF) to download them.

# Train
Take the Stanford Cars task in FGVC for example:
1. Replace '/path/to/FGVC/' with your path of the FGVC dataset in `train_scripts/vit/fgvc/stanford_cars.sh`
2. cd GPS
3. run
   `bash train_scripts/vit/fgvc/stanford_cars.sh`
   
# Cite
If this project is helpful for you, you can cite our paper:
```
@inproceedings{zhang2024gradient,
  title={Gradient-based Parameter Selection for Efficient Fine-Tuning},
  author={Zhang, Zhi and Zhang, Qizhe and Gao, Zijun and Zhang, Renrui and Shutova, Ekaterina and Zhou, Shiji and Zhang, Shanghang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={28566--28577},
  year={2024}
}
```


# Acknowledgement
Our experiment follows [SSF](https://github.com/dongzelian/SSF).

The code is built upon [SSF](https://github.com/dongzelian/SSF) and [VPT](https://github.com/KMnP/vpt).


