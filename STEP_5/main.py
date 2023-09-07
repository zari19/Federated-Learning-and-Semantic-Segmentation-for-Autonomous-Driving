import os
import json
import torch
import random
import numpy as np
from torchvision.models import resnet18
from utils.print_stats import print_stats
import datasets.ss_transforms as sstr
from datasets.cityscapes2 import CityScapesDataset
from torch import nn
from client import Client
from datasets.femnist import Femnist
from server import Server
from utils.args import get_parser
from datasets.idda import IDDADataset
from models.deeplabv3 import deeplabv3_mobilenetv2
from utils.stream_metrics import StreamSegMetrics, StreamClsMetrics
from utils.utils import setup_env

device = torch.device( 'cuda' if torch. cuda. is_available () else 'cpu')
import torch.optim.lr_scheduler as lr_scheduler
from utils.utils import HardNegativeMining, MeanReduction
from torch.utils.data import DataLoader
from torch import optim, nn
import matplotlib.pyplot as plt
from utils.yolo_seg import upscale_matrix, compare_arrays2, get_neighbors_new, map_and_insert, update_dictionary_values, update_dictionary_keys, get_class_map, map_values, process_matrix, get_result_matrix, process_dictionary, create_dict, pick_mask, pick_first_elem, yolo_prediction, yolo_model, merge_matrices, create_dict_probs, average_dictionary_values

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


def get_transforms(args): 
    if args.model == 'deeplabv3_mobilenetv2':
        train_transforms = sstr.Compose([
            sstr.RandomResizedCrop((512, 928), scale=(0.5,2.0)),
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
        root = "/root/idda"  #maybe change path
        root_cityscape = "/root/cityscape"
        with open(os.path.join(root, 'train.json'), 'r') as f:
            all_data = json.load(f)
        for client_id in all_data.keys():
            train_datasets.append(IDDADataset(root=root, list_samples=all_data[client_id], transform=train_transforms,
                                              client_name=client_id))
                                        
        with open(os.path.join(root, 'test_same_dom.txt'), 'r') as f:
            flag = False
            test_same_dom_data = f.read().splitlines()
            test_same_dom_dataset = IDDADataset(root=root, list_samples=test_same_dom_data, transform=test_transforms,
                                                client_name='test_same_dom')
        with open(os.path.join(root, 'test_diff_dom.txt'), 'r') as f:
            flag = False
            test_diff_dom_data = f.read().splitlines()
            test_diff_dom_dataset = IDDADataset(root=root, list_samples=test_diff_dom_data, transform=test_transforms,
                                                client_name='test_diff_dom')

        with open(os.path.join(root_cityscape, 'train.txt'), 'r') as f:
            test_diff_dom_data = f.read().splitlines()
            cityscape_dataset = CityScapesDataset(root=root_cityscape, list_samples=test_diff_dom_data, transform=test_transforms)

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

    return train_datasets, test_datasets, cityscape_dataset


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

def get_yolo_prediction(yolo_model, image_list, file_names_without_extensions):
        yolo_pred = yolo_prediction(model = yolo_model, image = image_list)

        yolo_masks = []
        yolo_prob_mask = []
                
        folder_path = "/root/runs/segment/predict/labels/"
        counter = 0
        for filename in file_names_without_extensions:
                        file_path = os.path.join(folder_path, filename)
                        check_txt_presence = os.path.join(folder_path, filename + ".txt")
                        flag_file = False
                        if not os.path.exists(os.path.join(folder_path, check_txt_presence)):

                            print(f'file {filename} not present... creating it')
                            flag_file = True
                            txt_path = os.path.join(folder_path, filename + ".txt")

                            with open(txt_path, 'w') as f:
                                f.write("79 0.1 0.1 0.1 0.1 0.1 0.1")
                        
                        txt_path = os.path.join(folder_path, filename + ".txt")
                        first_elem = pick_first_elem(txt_path)
                        class_map = get_class_map()
                        masks = pick_mask(first_elem, yolo_pred, counter, flag_file)
                        dict_probs = create_dict_probs(first_elem, yolo_pred[counter].boxes.conf)
                        avg_dict_prob = average_dictionary_values(dict_probs)
                        avg_dict_prob = update_dictionary_keys(avg_dict_prob, class_map)
                        avg_dict_prob = {key: value.item() for key, value in avg_dict_prob.items()}
                        dict_matrix = create_dict(first_elem, masks)
                        output_dict = process_dictionary(dict_matrix)
                        result_matrix = get_result_matrix(output_dict)
                        result_matrix = map_values(result_matrix, class_map)
                        new_matrix = upscale_matrix(result_matrix[0,:,:], (1080,1920))
                    
                        counter = counter +1
        print("Yolo prediction ended")
        return yolo_masks, yolo_prob_mask


def _configure_optimizer(args, params, current_round):
    if args.optimizer == 'SGD':
        optimizer = optim.SGD(params, lr=args.lr, momentum=args.m,
                              weight_decay=args.wd)
    elif args.optimizer == 'other':
        optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.wd)
        
    lr_lambda = lambda epoch: polynomial_decay(args, current_round, epoch)
    scheduler = lr_scheduler.StepLR(opt, step_size=5, gamma=0.1)

    return optimizer, scheduler

def _get_outputs(args, model, images):
        if args.model == 'deeplabv3_mobilenetv2':
            return model(images)['out']
        if args.model == 'resnet18':
            return model(images)
        raise NotImplementedError


def test(args, test_loader, metric, model, yolo_masks, yolo_prob_mask):
        criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='none')
        reduction = HardNegativeMining() if args.hnm else MeanReduction()
        """
        This method tests the model on the local dataset of the client.
        :param metric: StreamMetric object
        """
        print("Test...")
        model.eval()
        class_loss = 0.0
        ret_samples = []

        with torch.no_grad():
            for i, sample in enumerate(test_loader):
              images, labels = sample
              
              images = images.to(device, dtype=torch.float32)
              labels = labels.to(device, dtype=torch.long)

              outputs = _get_outputs(args, model, images)


              loss = reduction(criterion(outputs, labels),labels)
              class_loss += loss.item()
              _, prediction = outputs.max(dim=1)


              labels = labels.cpu().numpy()
              prediction = prediction.cpu().numpy()

              softmax = torch.nn.Softmax(dim=0)
              probability_matrix = softmax(outputs).cpu().numpy()
              pred2d = prediction[0,:,:]
              final_matrix = compare_arrays2(pred2d, yolo_masks[i], probability_matrix[0,0,:,:], yolo_prob_mask[i])

              metric['test_same_dom'].update(labels[0,:,:], final_matrix)

            class_loss = torch.tensor(class_loss).to(device)
            class_loss = class_loss / len(test_loader)

        return class_loss, ret_samples


def main():
    
    parser = get_parser() 
    args = parser.parse_args()  
    set_seed(args.seed) 
    reduction = HardNegativeMining() if args.hnm else MeanReduction()
    last_scores = defaultdict(lambda: defaultdict(lambda: []))


    print(f'Initializing model...')
    model = model_init(args)  #select type of model from the comand above
    
    model.cuda()
    yolov8_model = yolo_model()
    image_path_template = "/root/cityscapes/{}"  # Template for image paths
    image_list = []

    with open("/root/cityscape/train.txt", "r") as file:
        for line in file:
            image_name = line.strip()  # Remove newline characters
            image_path = image_path_template.format(image_name)
            image_list.append(image_path)

    file_names_without_extensions = []

    for file_path in image_list:
        file_name = os.path.basename(file_path) 
        file_name_without_extension, _ = os.path.splitext(file_name)  
        file_names_without_extensions.append(file_name_without_extension)

    print('Done.')
    
    print('Generate datasets...')
    train_datasets, test_datasets, cityscape_dataset = get_datasets(args)
    print('Done.')

    metrics = set_metrics(args)
    train_clients, test_clients = gen_clients(args, train_datasets, test_datasets, model)
    cityscape_dataloader = DataLoader(cityscape_dataset, batch_size=args.bs, shuffle=False)
    server = Server(args, train_clients, test_clients, model, metrics)
    PATH = "root/checkpoint"
    if not args.load_checkpoint:
        for r in tqdm(range(args.num_rounds)):
            
            loss, optimizer = server.train(metrics['eval_train'], r)
            print( f"ROUND {r + 1}/{args.num_rounds}: Training {args.clients_per_round} Clients...")
            
            train_score = metrics['eval_train'].get_results()
            print(train_score)

            round_losses = weight_train_loss(loss)
            print(round_losses)
            for name, l in round_losses.items():
                        print(f"R-{name}: {l}, step={r + 1}")
            metrics['eval_train'].reset()
            server.update_model()
    
        

    print("Train completed")
    if args.save_checkpoints == True:
                    PATH = "root/checkpoints/model_step5-1.pt"
                    LOSS = loss
                    checkpoint = {
                          'model_state_dict': model.state_dict(),
                          'optimizer_state_dict': optimizer.state_dict(),
                          'epoch': r,  # current epoch
                          # Add any other information you want to save
                      }
                    torch.save(checkpoint, PATH)
                  
                    print("checkpoint saved")

    if args.load_checkpoint:
                  PATH = "root/checkpoints/model_step5-1.pt"
                  print("\nloading checkpoints...")
                  opt = optim.SGD(model.parameters(), lr=args.lr, momentum=args.m,
                                      weight_decay=args.wd)
                  checkpoint = torch.load(PATH)
                  model.load_state_dict(checkpoint['model_state_dict'])
                  opt.load_state_dict(checkpoint['optimizer_state_dict'])
                  epoch = checkpoint['epoch']
                
                  model.train()
                  print("done")

    yolo_masks, yolo_prob_mask = get_yolo_prediction(yolov8_model, image_list, file_names_without_extensions)
    test(args, cityscape_dataloader, metrics, model, yolo_masks, yolo_prob_mask)
    test_score = metrics['test_same_dom'].get_results()

    print(test_score)

    print("Job completed!!")


if __name__ == '__main__':
    main()
