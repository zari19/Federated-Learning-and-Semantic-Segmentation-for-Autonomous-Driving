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


class IDDADataset(VisionDataset):

    def __init__(self,
                 root: str,
                 list_samples: [str],
                 transform: tr.Compose = None,
                 client_name: str = None):
        super().__init__(root=root, transform=transform, target_transform=None)
        self.list_samples = list_samples
        self.client_name = client_name
        self.target_transform = self.get_mapping()

    @staticmethod
    def get_mapping():
        classes = class_eval
        mapping = np.zeros((256,), dtype=np.int64) + 255
        for i, cl in enumerate(classes):
            mapping[i] = cl
        return lambda x: from_numpy(mapping[x])

    def __getitem__(self, index: int) -> Any: #called when used []
       
        sample_name = self.list_samples[index]

        root2_img = "/content/drive/MyDrive/idda/images"
        root2_labels = "/content/drive/MyDrive/idda/labels"
        image_path = os.path.join(root2_img, sample_name + ".jpg")
        label_path = os.path.join(root2_labels, sample_name + ".png")

        image=Image.open(self.root+'/images/'+self.list_samples[index]+'.jpg')
        target=Image.open (self.root + '/labels/'+self.list_samples[index]+".png")

        #1080x10920 is correct
        #data augmentation 
        if self.transform:
            image, label = self.transform(image,target)
            #image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(target)


        #print("idda bithh")
        return image, label

    def __len__(self) -> int:
        return len(self.list_samples)