import torch.nn as nn
from utils.stream_metrics import StreamSegMetrics
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import defaultdict
import torch
from utils.dist_utils import initialize_distributed
import wandb
from utils.logger import CustomWandbLogger, get_job_name, get_group_name, get_project_name
from utils.data_utils import Label2Color, color_map, Denormalize

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


def setup_pre_training(num_classes, dataset):
    train_metrics, val_metrics = set_metrics(num_classes, num_classes)
    ckpt_path = os.path.join('checkpoints', dataset)
    if not os.path.exists(ckpt_path):
        ckpt_path = os.makedirs(ckpt_path)
    if not os.path.exists(os.path.join(ckpt_path +"_bn")):
        os.makedirs(os.path.join(ckpt_path + "_bn"))
    else:
        ckpt_path = os.path.join('checkpoints', dataset)
    
    return train_metrics, val_metrics, ckpt_path
  
def load_from_checkpoint(dataset, ckpt_path, model, optimizer=None, scheduler=None):
    print("--- Loading model from checkpoint ---")
    load_path = os.path.join(ckpt_path + '.ckpt')
    print(load_path)
    checkpoint = torch.load(load_path)
    print(checkpoint)
    model.load_state_dict(checkpoint["model_state"])
    # if framework == 'federated':
    #     checkpoint_step = checkpoint["round"]
    #     writer.write(f"[!] Model restored from round {checkpoint['round']}")
    #     if 'last_scores' in checkpoint.keys():
    #         last_scores = checkpoint['last_scores']
    #         writer.write("Done.")
    #         return checkpoint_step, last_scores
    # elif framework == 'centralized':
    checkpoint_step = checkpoint["epoch"]
    scheduler.load_state_dict(checkpoint["scheduler_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    print(f"[!] Model restored from epoch {checkpoint['epoch']}")
    print("Done.")

    return checkpoint_step, None


def weight_train_loss(losses):
    """Function that weights losses over train round, taking only last loss for each user"""
    fin_losses = {}
    c = list(losses.keys())[0]
    loss_names = list(losses[c]['loss'].keys())
    for l_name in loss_names:
        tot_loss = 0
        weights = 0
        for _, d in losses.items():
            tot_loss += d['loss'][l_name][-1] * d['num_samples']
            weights += d['num_samples']
        fin_losses[l_name] = tot_loss / weights
    return fin_losses


def weight_test_loss(losses):
    tot_loss = 0
    weights = 0
    for k, v in losses.items():
        tot_loss = tot_loss + v['loss'] * v['num_samples']
        weights = weights + v['num_samples']
    return tot_loss / weights


def set_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True)


def setup_env(args):

    # ===== Setup distributed ===== #
    #device, rank, world_size = initialize_distributed(args.device_ids, args.local_rank)

    # ===== Initialize wandb ===== #

    if args.load:
        ids = args.wandb_id
    else:
        ids = wandb.util.generate_id()
        args.wandb_id = ids

    # logger = CustomWandbLogger(name=get_job_name(args), project=get_project_name(args.framework, args.dataset),
    #                            group=get_group_name(args), entity=args.wandb_entity, offline=args.wandb_offline,
    #                            resume="allow", wid=ids)
    # logger.log_hyperparams(args)


    # ===== Setup random seed to reproducibility ===== #
    if args.random_seed is not None:
        set_seed(args.random_seed)

    # ====== UTILS for ret_samples ===== #
    label2color = Label2Color(cmap=color_map(args.dataset, args.remap))  # convert labels to images
    if args.dataset == 'idda':
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    elif args.dataset == 'cityscapes':
        if args.cts_norm:
            mean = [0.3257, 0.3690, 0.3223]
            std = [0.2112, 0.2148, 0.2115]
        else:
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]
    else:
        mean, std = None, None
    denorm = Denormalize(mean=mean, std=std)  # de-normalization for original images

    return label2color, denorm

def get_scheduler(opts, optimizer, max_iter=None):
    if opts.lr_policy == 'poly':
        assert max_iter is not None, "max_iter necessary for poly LR scheduler"
        return torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                 lr_lambda=lambda cur_iter: (1 - cur_iter / max_iter) ** opts.lr_power)
    if opts.lr_policy == 'step':
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.lr_decay_step, gamma=opts.lr_decay_factor)

    return None


def print_images(value):
    if isinstance(value, np.ndarray):
        print("The value is a NumPy array.")
        plt.imshow(value)
        plt.axis('off')
        plt.show()
    elif isinstance(value, torch.Tensor):
        print("The value is a PyTorch tensor.")
    else:
        print("The value is neither a NumPy array nor a PyTorch tensor.")


