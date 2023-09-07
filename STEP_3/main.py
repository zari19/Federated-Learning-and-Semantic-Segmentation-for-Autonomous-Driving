import os
import json
from collections import defaultdict
import torch.optim as optim
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from torchvision.models import resnet18
from utils.print_stats import print_stats
import datasets.ss_transforms as sstr
import datasets.np_transforms as nptr
from torch import nn
from client import Client
from datasets.femnist import Femnist
from server import Server
from utils.args import get_parser
from datasets.idda import IDDADataset
from datasets.gta import GTADataset
from models.deeplabv3 import deeplabv3_mobilenetv2
from utils.stream_metrics import StreamSegMetrics, StreamClsMetrics
from utils.utils import setup_env
from utils.client_utils import setup_clients
from tqdm import tqdm
device = torch.device( 'cuda' if torch. cuda. is_available () else 'cpu')
import torch.optim.lr_scheduler as lr_scheduler
from utils.utils import HardNegativeMining, MeanReduction
from torch.utils.data import DataLoader

def set_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_dataset_num_classes(dataset): #return dataset number of classes
    if dataset == 'idda':
        return 16
    if dataset == 'femnist':
        return 62
    raise NotImplementedError


def model_init(args): #selects the type of model
    if args.model == 'deeplabv3_mobilenetv2':
        return deeplabv3_mobilenetv2(num_classes=get_dataset_num_classes(args.dataset))
    if args.model == 'resnet18':
        model = resnet18()
        model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.fc = nn.Linear(in_features=512, out_features=get_dataset_num_classes(args.dataset))
        return model
    if args.model == 'cnn':
        raise NotImplementedError
    raise NotImplementedError


def get_transforms(args): #perform data augmentation based on the model
   
    if args.model == 'deeplabv3_mobilenetv2':
        train_transforms = sstr.Compose([
            sstr.RandomResizedCrop((512, 928), scale=(.5, 2.0)),
            sstr.ToTensor(),
            sstr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        test_transforms = sstr.Compose([
            sstr.ToTensor(),
            sstr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif args.model == 'cnn' or args.model == 'resnet18':
        train_transforms = nptr.Compose([
            nptr.ToTensor(),
            nptr.Normalize((0.5,), (0.5,)),
        ])
        test_transforms = nptr.Compose([
            nptr.ToTensor(),
            nptr.Normalize((0.5,), (0.5,)),
        ])
    else:
        raise NotImplementedError
    return train_transforms, test_transforms


def read_femnist_dir(data_dir):
    data = defaultdict(lambda: {})
    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.json')]
    for f in files:
        file_path = os.path.join(data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        data.update(cdata['user_data'])
    return data


def read_femnist_data(train_data_dir, test_data_dir):
    return read_femnist_dir(train_data_dir), read_femnist_dir(test_data_dir)


def get_gta(args):
    if args.dataset == 'idda':
        train_transforms, test_transforms = get_transforms(args)
        root = "/root/GTA5/" 
        root_gta_img = "/root/GTA5/images"
        root_gta_labels = "/root/GTA5/labels"
        with open(os.path.join(root, 'train_gta.txt'), 'r') as f:
            flag = True
            train_gta = f.read().splitlines()
            gta_dataset = GTADataset(root=root, list_samples=train_gta, transform=train_transforms, flag =flag)
    return gta_dataset

def get_datasets(args): #get access to datasets in root/idda

    train_datasets = []
    train_transforms, test_transforms = get_transforms(args)

    if args.dataset == 'idda':
        root = "/root/idda" 
        root_gta_img = "/root/GTA5/images"
        root_gta_labels = "/root/GTA5/labels"


        with open(os.path.join(root, 'train_gta.txt'), 'r') as f:

            train_gta = f.read().splitlines()
            gta_dataset = GTADataset(root=root, list_samples=train_gta, transform=train_transforms, flag =flag)

        with open(os.path.join(root, 'train.txt'), 'r') as f:

            eval_idda = f.read().splitlines()
            eval_idda_dataset = IDDADataset(root=root, list_samples=eval_idda, transform=train_transforms)
                                        
        with open(os.path.join(root, 'test_same_dom.txt'), 'r') as f:

            test_same_dom_data = f.read().splitlines()

            test_same_dom_dataset = IDDADataset(root=root, list_samples=test_same_dom_data, transform=test_transforms)
        with open(os.path.join(root, 'test_diff_dom.txt'), 'r') as f:

            test_diff_dom_data = f.read().splitlines()
            test_diff_dom_dataset = IDDADataset(root=root, list_samples=test_diff_dom_data, transform=test_transforms,
                                                client_name='test_diff_dom')

        with open(os.path.join(root, 'test_join.txt'), 'r') as f:

            test_join = f.read().splitlines()
            test_join_dataset = IDDADataset(root=root, list_samples=test_join, transform=test_transforms,
                                                client_name='test_join')
        test_datasets = [test_same_dom_dataset, test_diff_dom_dataset, test_join_dataset]

    elif args.dataset == 'femnist':
        niid = args.niid
        train_data_dir = os.path.join('data', 'femnist', 'data', 'niid' if niid else 'iid', 'train')
        test_data_dir = os.path.join('data', 'femnist', 'data', 'niid' if niid else 'iid', 'test')
        train_data, test_data = read_femnist_data(train_data_dir, test_data_dir)

        train_transforms, test_transforms = get_transforms(args)

        train_datasets, test_datasets = [], []

        for user, data in train_data.items():
            train_datasets.append(Femnist(data, train_transforms, user))
        for user, data in test_data.items():
            test_datasets.append(Femnist(data, test_transforms, user))

    else:
        raise NotImplementedError

    return train_datasets, test_datasets, eval_idda_dataset


def set_metrics(args):
    num_classes = get_dataset_num_classes(args.dataset)
    if args.model == 'deeplabv3_mobilenetv2':
        metrics = {
            'eval_train': StreamSegMetrics(num_classes, 'eval_train'),
            'test_same_dom': StreamSegMetrics(num_classes, 'test_same_dom'),
            'test_diff_dom': StreamSegMetrics(num_classes, 'test_diff_dom')
        }
    elif args.model == 'resnet18' or args.model == 'cnn':
        metrics = {
            'eval_train': StreamClsMetrics(num_classes, 'eval_train'),
            'test': StreamClsMetrics(num_classes, 'test')
        }
    else:
        raise NotImplementedError
    return metrics


def gen_clients(args, train_datasets, test_datasets, model):
    clients = [[], []]
    for i, datasets in enumerate([train_datasets, test_datasets]):
        for ds in datasets:
            clients[i].append(Client(args, ds, model, test_client=i == 1))
    return clients[0], clients[1]


def main():
    parser = get_parser()
    args = parser.parse_args() 
    set_seed(args.seed)

    reduction = HardNegativeMining() if args.hnm else MeanReduction()

    def get_optimizer(net, lr, wd, momentum):
      optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=wd, momentum=momentum)
      return optimizer

    def _get_outputs(images):
        if args.model == 'deeplabv3_mobilenetv2':
            return model(images)['out']
        if args.model == 'resnet18':
            return model(images)
        raise NotImplementedError


    def update_metric(metrics, outputs, labels, cur_step):
        _, prediction = outputs.max(dim=1)
        labels = labels.cpu().numpy()
        prediction = prediction.cpu().numpy()
        metrics.update(labels, prediction)

    def load_checkpoints(PATH):
        checkpoint = torch.load(PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        opt.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']


    def model_eval(metric):
        print("Model evaluation...")
        model.eval()

        for cur_step, (images, labels) in (enumerate(eval_dataloader)):

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            outputs = _get_outputs(images)
            _, prediction = outputs.max(dim=1)
            labels = labels.cpu().numpy()
            prediction = prediction.cpu().numpy()
            metric['eval_train'].update(labels, prediction)

        
    def test(metric):
        """
        This method tests the model on the local dataset of the client.
        :param metric: StreamMetric object
        """
        print("Testing...")
        model.eval()
        class_loss = 0.0
        ret_samples = []

        with torch.no_grad():
            for i, sample in (enumerate(test_join_dataloader)):
              images, labels = sample
              
              images = images.to(device, dtype=torch.float32)
              labels = labels.to(device, dtype=torch.long)

              outputs = _get_outputs(images)

              loss = reduction(criterion(outputs, labels),labels)
              class_loss += loss.item()

              _, prediction = outputs.max(dim=1)
              labels = labels.cpu().numpy()
              prediction = prediction.cpu().numpy()

              metric['test_same_dom'].update(labels, prediction)

            class_loss = torch.tensor(class_loss).to(device)
            print(f'class_loss = {class_loss}')
            class_loss = class_loss / len(test_join_dataloader)

        return class_loss, ret_samples


    print("Step 3")
    print(f'Initializing model...')
    model = model_init(args)
    model.cuda()
    print('Done.')

    print('Generate datasets...')

    train_datasets, test_datasets, eval_dataset = get_datasets(args)
    gta_dataset = get_gta(args)
    gta_dataloader = DataLoader(gta_dataset, batch_size=args.bs, shuffle=False)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.bs, shuffle=False)
    test_join_dataloader = DataLoader(test_datasets[2], batch_size=args.bs, shuffle=False)

    metrics = set_metrics(args)

    if args.load != "True":
        model.train()
        print("Training...")
        net = model
        opt = get_optimizer(net, lr=args.lr, wd=args.wd, momentum=args.m)
        scheduler = lr_scheduler.StepLR(opt, step_size=5, gamma=0.1)
        for r in tqdm(range(args.num_epochs), total= args.num_epochs):

        
            dict_all_epoch_losses = defaultdict(lambda: 0)
            running_loss = 0.0

            for cur_step, (images, labels) in (enumerate(gta_dataloader)):

                images = images.to(device, dtype=torch.float32)
                labels = labels.to(device, dtype=torch.long)

                opt.zero_grad()
                outputs = _get_outputs(images)
                criterion = nn.CrossEntropyLoss(ignore_index=255,reduction='none')
                loss = reduction(criterion(outputs, labels), labels)
                loss.backward()
                opt.step()
                running_loss += loss.item()
            
            epoch_loss = running_loss / len(gta_dataloader)
            print(f'Epoch [{r+1}/{args.num_epochs}], Loss: {epoch_loss:.4f}')

            scheduler.step()

        print("Train completed")

    if args.load == "True":
        load_checkpoints(PATH = "checkpoint/model.pt")

    model_eval(metrics)
    eval_score = metrics['eval_train'].get_results()
    print(eval_score) 

    test( metrics)
    test_score = metrics['test_same_dom'].get_results()
    print(test_score) 
      
    print("Job completed!!")


if __name__ == '__main__':
    main()
