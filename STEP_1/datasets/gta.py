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
    
        

class_eval = [255, 2, 4, 255, 11, 5, 0, 0, 1, 8, 13, 3, 7, 6, 255, 255, 15, 14, 12, 9, 10]


class GTADataset(VisionDataset):
    def __init__(self, root: str, list_samples: [str], transform: tr.Compose = None, 
                 flag=False, use_mapping=True):
        super().__init__(root=root, transform=transform, target_transform=None)
        self.list_samples = list_samples
        self.target_transform = self.get_mapping() if use_mapping else None
        self.flag = flag

    @staticmethod
    def get_mapping():
        class_map = {
            1: 13,  # ego_vehicle : vehicle
            7: 0,   # road
            8: 1,   # sidewalk
            11: 2,  # building
            12: 3,  # wall
            13: 4,  # fence
            17: 5,  # pole
            18: 5,  # poleGroup: pole
            19: 6,  # traffic light
            20: 7,  # traffic sign
            21: 8,  # vegetation
            22: 9,  # terrain
            23: 10,  # sky
            24: 11,  # person
            25: 12,  # rider
            26: 13,  # car : vehicle
            27: 13,  # truck : vehicle
            28: 13,  # bus : vehicle
            32: 14,  # motorcycle
            33: 15,  # bicycle
        }

        def map_labels(label):
            label_np = np.array(label)
            mapped_label = np.zeros(label_np.shape, dtype=np.int64) + 255

            for key, value in class_map.items():
                mapped_label[label_np == key] = value

            return from_numpy(mapped_label)

        return map_labels

    def __getitem__(self, index: int) -> Any:
       
        if self.flag:
            #print("flag")
            sample_name = self.list_samples[index]
            root2_img = '/content/drive/MyDrive/data/GTA5/images/'
            root2_labels = "/content/drive/MyDrive/data/GTA5/labels/"
            image_path = os.path.join(root2_img, sample_name)
            label_path = os.path.join(root2_labels, sample_name)

            image = Image.open(root2_img + self.list_samples[index])
            target = Image.open(root2_labels + self.list_samples[index])
            # image.save('train_before_trasform.png')
            # target.save('target_before_trasform.png')

            if image.size != (1920, 1080):
                image = image.resize((1920, 1080))

            if target.size != (1920, 1080):
                target = target.resize((1920, 1080))
        else:
            print('idda')
            image = Image.open(self.root + '/images/' + self.list_samples[index] + '.jpg')
            target = Image.open(self.root + "/labels/" + self.list_samples[index] + ".png")
            use_mapping = False

        #print(f'before transform {np.unique(target)}')

        if self.transform:
            image, label = self.transform(image, target)

        if self.target_transform:
            label = self.target_transform(target)

        

        return image, label

    def __len__(self) -> int:
        return len(self.list_samples)