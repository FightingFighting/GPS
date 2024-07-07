#!/usr/bin/env python3

"""JSON dataset: support CUB, NABrids, Flower, Dogs and Cars"""

import os
import torch
import torch.utils.data
import torchvision as tv
import numpy as np
from collections import Counter
from torch.utils.data import Dataset

from utils.io_utils import read_json


class FGVC(Dataset):
    def __init__(self, root, name=None, train=True, transform=None, target_transform=None):
        if "cub" in name:
            root = os.path.join(root, "CUB_200_2011")
            img_dir = os.path.join(root, "images")
        elif "nabird" in name:
            root = os.path.join(root, "nabirds")
            img_dir = os.path.join(root, "images")
        elif "flower" in name:
            root = os.path.join(root, "oxfordflower")
            img_dir = root
        elif "dog" in name:
            root = os.path.join(root, "stanforddogs")
            img_dir = os.path.join(root, "Images")
        elif "car" in name:
            root = os.path.join(root, "stanfordcars")
            img_dir = root

        self.root = root
        self.img_dir = img_dir
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        
        self.anno = dict()
        self.split = dict()
        
        train_split = read_json(os.path.join(root, "train.json"))
        val_split = read_json(os.path.join(root, "val.json"))
        test_split = read_json(os.path.join(root, "test.json"))
        
        self.anno.update(train_split)
        self.anno.update(val_split)
        self.anno.update(test_split)
        if train:
            self.split.update(train_split)
            self.split.update(val_split)
        else:
            self.split.update(test_split)
        
        # Map class ids to contiguous ids
        self.class_ids = sorted(list(set(self.anno.values())))
        self.class_cont_ids = {v: i for i, v in enumerate(self.class_ids)}
        
        # Construct the image db
        self.data = []
        for img_name, cls_id in self.split.items():
            img_path = os.path.join(img_dir, img_name)
            cont_id = self.class_cont_ids[cls_id]
            self.data.append({"img_path": img_path, "class": cont_id})

        flag = "train" if train else "test"
        print("Number of {} images: {}".format(flag, len(self.data)))
        print("Number of {} classes: {}".format(flag, len(self.class_ids)))
            

    def __getitem__(self, index):
        image = tv.datasets.folder.default_loader(self.data[index]["img_path"])
        target = self.data[index]["class"]
        
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
            
        return image, target

    def __len__(self):
        return len(self.data)


class CUB200Dataset(FGVC):
    """CUB_200 dataset."""

    def __init__(self, cfg, split):
        super(CUB200Dataset, self).__init__(cfg, split)

    def get_imagedir(self):
        return os.path.join(self.data_dir, "images")


class CarsDataset(FGVC):
    """stanford-cars dataset."""

    def __init__(self, cfg, split):
        super(CarsDataset, self).__init__(cfg, split)

    def get_imagedir(self):
        return self.data_dir


class DogsDataset(FGVC):
    """stanford-dogs dataset."""

    def __init__(self, cfg, split):
        super(DogsDataset, self).__init__(cfg, split)

    def get_imagedir(self):
        return os.path.join(self.data_dir, "Images")


class FlowersDataset(FGVC):
    """flowers dataset."""

    def __init__(self, cfg, split):
        super(FlowersDataset, self).__init__(cfg, split)

    def get_imagedir(self):
        return self.data_dir


class NabirdsDataset(FGVC):
    """Nabirds dataset."""

    def __init__(self, cfg, split):
        super(NabirdsDataset, self).__init__(cfg, split)

    def get_imagedir(self):
        return os.path.join(self.data_dir, "images")

