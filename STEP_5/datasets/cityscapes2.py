from enum import Flag
import os
from typing import Any
import torch
import numpy as np
from PIL import Image
from torch import from_numpy
from torchvision.datasets import VisionDataset
import datasets.ss_transforms as tr
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
    
        

eval_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]


class CityScapesDataset(VisionDataset):
    def __init__(self, root: str, list_samples: [str], transform: tr.Compose = None, 
                 flag=False, use_mapping=True):
        super().__init__(root=root, transform=transform, target_transform=None)
        self.list_samples = list_samples
        self.target_transform = self.get_mapping() if use_mapping else None
        self.flag = flag

    @staticmethod
    def get_mapping():
        class_map = {
            7:0, # "road",  # 1
            8:1, # "sidewalk",  # 2
            9:0,# "parking",
            10:13, # "rail truck",
            11:2, # "building",  # 3
            12:3, # "wall",  # 4
            13:4, # "fence",  # 5
            14:4, # "guard_rail",
            15:2, # "bridge",
            16:2, # "tunnel",
            17:5, # "pole",  # 6
            18:5, # "pole_group",
            19:6, # "light",  # 7
            20:7, # "sign",  # 8
            21:8, # "vegetation",  # 9
            22:9, # "terrain",  # 10
            23:10, # "sky",  # 11
            24:11, # "person",  # 12
            25:12, # "rider",  # 13
            26:13, # "car",  # 14
            27:13, # "truck",  # 15
            28:13, # "bus",  # 16
            29:13,# "caravan",
            30:13, # "trailer",
            31:13, # "train",  # 17
            32:14, # "motorcycle",  # 18
            33:15, # "bicycle"  # 19
        }

        def map_labels(label):
            label_np = np.array(label)
            mapped_label = np.zeros(label_np.shape, dtype=np.int64) + 255

            for key, value in class_map.items():
                mapped_label[label_np == key] = value

            return from_numpy(mapped_label)

        return map_labels

    def __getitem__(self, index: int) -> Any:
       

            #print("flag")
            sample_name = self.list_samples[index]

            cityscape_root = "/content/drive/MyDrive/cityscape_fda/"
            cityscape_label = "/content/drive/MyDrive/data/Cityscapes/labels/"
            str = self.list_samples[index][:-15]
            to_add = "gtFine_labelIds.png"
            res = str+to_add
            res

            image = Image.open(cityscape_root + self.list_samples[index])
            target = Image.open(cityscape_label + res)
            # target.save('target_before_trasform.png')

            if image.size != (1920, 1080):
                image = image.resize((1920, 1080))

            if target.size != (1920, 1080):
                target = target.resize((1920, 1080))

            #print(f'before transform {np.unique(target)}')

            if self.transform:
                image, label = self.transform(image, target)

            if self.target_transform:
                label = self.target_transform(target)

            

            return image, label

    def __len__(self) -> int:
        return len(self.list_samples)