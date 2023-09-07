import os
import json
from collections import defaultdict
import torch
from torch.utils.data import DataLoader
import random
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
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
from models.deeplabv3 import deeplabv3_mobilenetv2
from utils.stream_metrics import StreamSegMetrics, StreamClsMetrics
from utils.utils import setup_env
from utils.client_utils import setup_clients
from tqdm import tqdm

device = torch.device( 'cuda' if torch. cuda. is_available () else 'cpu')


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
            sstr.RandomResizedCrop((512, 928), scale=(0.5, 2.0)),
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


def get_datasets(args): #get access to datasets in root/idda

    train_datasets = []
    train_transforms, test_transforms = get_transforms(args)

    if args.dataset == 'idda':
        root = "/root/idda"
        psudo_labels_root = "/root/pseudo_label"

        with open(os.path.join(root, 'train.json'), 'r') as f:
            all_data = json.load(f)
        for client_id in all_data.keys():
            train_datasets.append(IDDADataset(root=root, list_samples=all_data[client_id], transform=train_transforms,
                                              client_name=client_id))

        with open(os.path.join(root, 'train.txt'), 'r') as f:
            flag = True
            train_idda = f.read().splitlines()
            train_idda_dataset = IDDADataset(root=root, list_samples=train_idda, transform=train_transforms)
            

        with open(os.path.join(root, 'test_same_dom.txt'), 'r') as f:
            test_same_dom_data = f.read().splitlines()
            test_same_dom_dataset = IDDADataset(root=root, list_samples=test_same_dom_data, transform=test_transforms,
                                                client_name='test_same_dom')
        with open(os.path.join(root, 'test_diff_dom.txt'), 'r') as f:
            test_diff_dom_data = f.read().splitlines()
            test_diff_dom_dataset = IDDADataset(root=root, list_samples=test_diff_dom_data, transform=test_transforms,
                                                client_name='test_diff_dom')
        test_datasets = [test_same_dom_dataset, test_diff_dom_dataset]

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

    return train_datasets, train_idda_dataset, test_datasets


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

    def _get_outputs(images):
        if args.model == 'deeplabv3_mobilenetv2':
            return model(images)['out']
        if args.model == 'resnet18':
            return model(images)
        raise NotImplementedError

    def load_checkpoints(PATH):
        checkpoint = torch.load(PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']

    def get_pseudo_label(model):
        print("Model evaluation...")
        model.eval()

        for cur_step, (images, labels) in enumerate(idda_dataset):

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            outputs = _get_outputs(images)
            _, prediction = outputs.max(dim=1)
            labels = labels.cpu().numpy()
            prediction = prediction.cpu().numpy()
            pseudo_label = prediction[0,:,:]

    last_scores = defaultdict(lambda: defaultdict(lambda: []))

    print("Step 4... actually working")
    print(f'Initializing model...')
    model = model_init(args)
    model.cuda()
    print('Done.')

    print('Generate datasets...')
    train_datasets, train_idda_dataset, test_datasets = get_datasets(args)
    idda_dataset = DataLoader(train_idda_dataset, batch_size=args.bs, shuffle=False)
    print('Done.')

    metrics = set_metrics(args)
    train_clients, test_clients = gen_clients(args, train_datasets, test_datasets, model)
    server = Server(args, train_clients, test_clients, model, metrics)


    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.m)
    

    
    load_checkpoints(PATH = "/rooot/checkpoint/model.pt")

    student_model = model
    teacher_model = model
    
    t_rounds = 2
    for r in tqdm(range(args.num_rounds)):
            
            loss = server.train(metrics['eval_train'], student_model = student_model, teacher_model = teacher_model)
            print( f"ROUND {r + 1}/{args.num_rounds}: Training {args.clients_per_round} Clients...")

            train_score = metrics['eval_train'].get_results()
            print(train_score)

            round_losses = weight_train_loss(loss)
            print(round_losses)
            for name, l in round_losses.items():
                        print(f"R-{name}: {l}, step={r + 1}")
            metrics['eval_train'].reset()
            if args.modality4 == "2":
                print("modality = 2")
                server.update_model()
            print(f'round numero ={r}')
            if args.modality4 == "3" and r % t_rounds ==0:
                print("modality = 3")
                server.update_model()
           

    print("Train completed")

    server.test(test_clients, metrics, model)
    test_score = metrics['test_same_dom'].get_results()

    print(test_score) 

    print("Job completed!!")

if __name__ == '__main__':
    main()
