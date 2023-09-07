import torch.nn as nn
from utils.stream_metrics import StreamSegMetrics
import json
import os
import numpy as np
import random
from collections import defaultdict
import torch


class HardNegativeMining(nn.Module):

    def __init__(self, perc=0.25):
        super().__init__()
        self.perc = perc

    def forward(self, loss, _):
        b = loss.shape[0]
        loss = loss.reshape(b, -1)
        p = loss.shape[1]
        tk = loss.topk(dim=1, k=int(self.perc * p))
        loss = tk[0].mean()
        return loss


class MeanReduction:
    def __call__(self, x, target):
        x = x[target != 255]
        return x.mean()

def set_metrics(num_classes, name):
    print("Setting up metrics...")
    val_metrics = StreamSegMetrics(num_classes, name)
    train_metrics = StreamSegMetrics(num_classes, name)
    print("Done.")
    return train_metrics, val_metrics
