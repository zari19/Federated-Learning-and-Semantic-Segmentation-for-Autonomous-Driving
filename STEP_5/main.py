import os
import json
from collections import defaultdict
from utils.utils import setup_pre_training, load_from_checkpoint
import torch
import random
import numpy as np
from torchvision.models import resnet18
from utils.print_stats import print_stats
import datasets.ss_transforms as sstr
import datasets.np_transforms as nptr
from datasets.cityscapes import Cityscapes
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
from utils.client_utils import setup_clients
from time import sleep
from tqdm import tqdm
from google.colab import auth
import gspread
from google.auth import default
device = torch.device( 'cuda' if torch. cuda. is_available () else 'cpu')
import torch.optim.lr_scheduler as lr_scheduler
from utils.utils import HardNegativeMining, MeanReduction
from torch.utils.data import DataLoader
#from datasets.gta import GTADataset
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
        # TODO: missing code here!
        raise NotImplementedError
    raise NotImplementedError


def get_transforms(args): #perform data augmentation based on the model
    # TODO: test your data augmentation by changing the transforms here!
    if args.model == 'deeplabv3_mobilenetv2':
        train_transforms = sstr.Compose([
            #sstr.RandomResizedCrop((1920, 1080), scale=(1.0, 1.0)), #default  512, 928  #scale .5,2
            sstr.ToTensor(),
            #sstr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        test_transforms = sstr.Compose([
            sstr.ToTensor(),
            #sstr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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


def get_gta(args):
    if args.dataset == 'idda':
        train_transforms, test_transforms = get_transforms(args)
        root = "/content/drive/MyDrive/idda"  #maybe change path
        root_gta_img = "/content/drive/MyDrive/data/GTA/images"
        root_gta_labels = "/content/drive/MyDrive/data/GTA/labels"
        gta_root = "/content/drive/MyDrive/gta_001/"
        with open(os.path.join(root, 'train_gta.txt'), 'r') as f:
            flag = True
            train_gta = f.read().splitlines()
            gta_dataset = GTADataset(root=gta_root, list_samples=train_gta, transform=train_transforms, flag =flag)
    return gta_dataset

def read_femnist_data(train_data_dir, test_data_dir):
    return read_femnist_dir(train_data_dir), read_femnist_dir(test_data_dir)


def get_datasets(args): #get access to datasets in root/idda

    train_datasets = []
    train_transforms, test_transforms = get_transforms(args)

    if args.dataset == 'idda':
        root = "/content/drive/MyDrive/idda"  #maybe change path
        root_cityscape = "/content/drive/MyDrive/cityscape_fda"
        #/content/drive/MyDrive/idda/test_same_dom.txt
        #/content/drive/MyDrive/idda
        #/content/drive/MyDrive/MLDL/MLDL23-FL-project-main/data/idda
        flag = False
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

        with open(os.path.join(root_cityscape, 'train_half.txt'), 'r') as f:
            test_diff_dom_data = f.read().splitlines()
            data_list = [{'x': item, 'y': item.replace('leftImg8bit', 'gtFine_labelIds')} for item in test_diff_dom_data]

            print(test_diff_dom_data)
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
                
        folder_path = "/content/drive/MyDrive/MLDL23-FL-step5-fda-yolo5/runs/segment/predict/labels/"
        counter = 0
        for filename in file_names_without_extensions:
                        file_path = os.path.join(folder_path, filename)
                        
                        #print(filename)
                        # print(file_path)
                        check_txt_presence = os.path.join(folder_path, filename + ".txt")
                        #print(check_txt_presence)
                        flag_file = False
                        if not os.path.exists(os.path.join(folder_path, check_txt_presence)):

                            print(f'file {filename} not present... creating it')
                            flag_file = True
                            txt_path = os.path.join(folder_path, filename + ".txt")

                                # Create an empty text file
                            with open(txt_path, 'w') as f:
                                f.write("79 0.1 0.1 0.1 0.1 0.1 0.1")
                        
                        txt_path = os.path.join(folder_path, filename + ".txt")
                        first_elem = pick_first_elem(txt_path)
                        #print(first_elem)
                        #print(flag_file)
                        class_map = get_class_map()
                        masks = pick_mask(first_elem, yolo_pred, counter, flag_file)
                        # print("\navg dict probs")
                        # print(first_elem, yolo_pred[0].boxes.conf)
                        dict_probs = create_dict_probs(first_elem, yolo_pred[counter].boxes.conf)
                        avg_dict_prob = average_dictionary_values(dict_probs)
                        avg_dict_prob = update_dictionary_keys(avg_dict_prob, class_map)
                        avg_dict_prob = {key: value.item() for key, value in avg_dict_prob.items()}
                        # print("\navg dict probs")
                        # print(avg_dict_prob)
                        dict_matrix = create_dict(first_elem, masks)
                        output_dict = process_dictionary(dict_matrix)
                        result_matrix = get_result_matrix(output_dict)
                        
                        result_matrix = map_values(result_matrix, class_map)
                        plt.imshow(result_matrix[0,:,:])
                        plt.savefig('masks/pred{}.png'.format(counter))
                        #remapped_array = np.vectorize(class_map.get)(result_matrix)
                        # print("\nresult matrix")
                        # print(np.unique(result_matrix, return_counts=True))
                        new_matrix = upscale_matrix(result_matrix[0,:,:], (1080,1920))
                        

                        if counter %2==0:
                            yolo_masks.append(new_matrix)
                            # print("\nyolo mask")
                            # print(np.unique(new_matrix, return_counts=True))
                            mtr = map_and_insert(new_matrix, avg_dict_prob)
                            # print("\n yolo part")
                            # print(f'yolo_prob = {np.unique(mtr, return_counts=True)}')
                            # print(f'yolo_pred = {np.unique(new_matrix, return_counts=True)}')
                            # print("\nyolo pred")
                            # print(np.unique(mtr, return_counts=True))
                            #mtr[np.where(mtr == None)] = -1
                            yolo_prob_mask.append(mtr)
                            #print(yolo_prob_mask)

                        counter = counter +1
        print("Yolo prediction ended")
        return yolo_masks, yolo_prob_mask

def get_flag():
        x=0
        print("porca madonna")
        return x

def get_optimizer(net, lr, wd, momentum):
      optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=wd, momentum=momentum)
      return optimizer

def _configure_optimizer(args, params, current_round):
    if args.optimizer == 'SGD':
        optimizer = optim.SGD(params, lr=args.lr, momentum=args.m,
                              weight_decay=args.wd)
    elif args.optimizer == 'other':
        optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.wd)

    # Define the learning rate lambda function for PolynomialLR scheduler
    lr_lambda = lambda epoch: polynomial_decay(args, current_round, epoch)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    return optimizer, scheduler

def polynomial_decay(args, current_round, epoch):
    initial_lr = 0.1
    final_lr = 0.0001
    power = 2
    decay_factor = (1 - current_round / args.num_epochs) ** power
    lr = initial_lr + (final_lr - initial_lr) * decay_factor
    return lr

def _get_outputs(args, model, images):
        if args.model == 'deeplabv3_mobilenetv2':
            return model(images)['out']
        if args.model == 'resnet18':
            return model(images)
        raise NotImplementedError


def train_gta(args, model, gta_dataloader):
        reduction = HardNegativeMining() if args.hnm else MeanReduction()
        model.train()
        print("Training...")
        net = model
        
        for r in tqdm(range(args.num_epochs), total= args.num_epochs):
            opt, scheduler = _configure_optimizer(args,net.parameters(), r)
            #loss = server.train(metrics['eval_train'])
            dict_all_epoch_losses = defaultdict(lambda: 0)
            running_loss = 0.0

            for cur_step, (images, labels) in (enumerate(gta_dataloader)):

                images = images.to(device, dtype=torch.float32)
                labels = labels.to(device, dtype=torch.long)
   
                opt.zero_grad()
                outputs = _get_outputs(args, model, images)
           
                criterion = nn.CrossEntropyLoss(ignore_index=255,reduction='none')
                loss = reduction(criterion(outputs, labels), labels)
                loss.backward()
                opt.step()
                running_loss += loss.item()
            
            epoch_loss = running_loss / len(gta_dataloader)
            print(f'Epoch [{r+1}/{args.num_epochs}], Loss: {epoch_loss:.4f}')

def test2(args, test_loader, metric, model, yolo_masks, yolo_prob_mask):
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

              #get prediction for each pixel
              # softmax = torch.nn.Softmax(dim=1)
              # probs = softmax(outputs)
              # predicted_labels = torch.argmax(outputs, dim=1)
              # max_probs, _ = torch.max(probs, dim=1)
              # predicted_labels = predicted_labels.squeeze(0).cpu().numpy()
              # max_probs = max_probs.squeeze(0).cpu().numpy()
              # #print(max_probs.shape)
              # prediction_array = max_probs[0,:,:]
              # #print(predicted_labels)

              #pred_labels = outputs.argmax(1)[0].cpu().numpy()

              

              #yolo step =======================================
              softmax = torch.nn.Softmax(dim=0)
              probability_matrix = softmax(outputs).cpu().numpy()
              pred2d = prediction[0,:,:]
              # if i == 2:

              #     print(pred2d.shape)
              #     print(probability_matrix.shape)
              #     print(yolo_masks[i].shape)
              #     print(yolo_prob_mask[i].shape)
              #     #np.save("pred_deep_lab", pred2d)
              #     #np.save("deep_lab_probs", probability_matrix[0,0,:,:])
              #     # np.save("yolo_pred", yolo_masks[i])
              #     # np.save("yolo_probs", yolo_prob_mask[i])

              # print("\ntest values ")
              # print(f'pred_deep_lab = {np.unique(pred2d, return_counts=True)}')
              # print(f'probs_deep_lab = {np.unique(probability_matrix[0,0,:,:], return_counts=True)}')
              # print(f'preds_yolo = {np.unique(yolo_masks[i], return_counts=True)}')
              # print(f'probs_yolo = {np.unique(yolo_prob_mask[i], return_counts=True)}')


              final_matrix = compare_arrays2(pred2d, yolo_masks[i], probability_matrix[0,0,:,:], yolo_prob_mask[i])

              metric['test_same_dom'].update(labels[0,:,:], final_matrix)

              pred2 = final_matrix
              plt.imshow(pred2)
              plt.savefig('yolo_combined_pred/pred{}.png'.format(i))

              #======================================

              # metric['test_same_dom'].update(labels, prediction)
              # pred2 = prediction[0,:,:]
              # plt.imshow(pred2)
              # plt.savefig('test_imgs/pred{}.png'.format(i))

            class_loss = torch.tensor(class_loss).to(device)
            # print(f'class_loss = {class_loss}')
            # print(f'len labels {len(labels)}')
            # print(f'len final_matrix {len(final_matrix)}')
            class_loss = class_loss / len(test_loader)

        return class_loss, ret_samples


def main():
    
    parser = get_parser() #calls function inside utils.args, define seed, #clients ecc.
    args = parser.parse_args()  #??
    set_seed(args.seed) #??
    reduction = HardNegativeMining() if args.hnm else MeanReduction()

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

    last_scores = defaultdict(lambda: defaultdict(lambda: []))


    print(f'Initializing model...')
    model = model_init(args)  #select type of model from the comand above
    
    model.cuda()
    yolov8_model = yolo_model()
    image_path_template = "/content/drive/MyDrive/cityscape_fda/{}"  # Template for image paths
    image_list = []

    with open("/content/drive/MyDrive/cityscape_fda/train_half.txt", "r") as file:
        for line in file:
            image_name = line.strip()  # Remove newline characters
            image_path = image_path_template.format(image_name)
            image_list.append(image_path)

    #print(image_list)
    file_names_without_extensions = []

    for file_path in image_list:
        file_name = os.path.basename(file_path)  # Get the file name with extension
        file_name_without_extension, _ = os.path.splitext(file_name)  # Split file name and extension
        file_names_without_extensions.append(file_name_without_extension)

    #print(file_names_without_extensions)

    print('Done.')
    
    print('Generate datasets...')
    train_datasets, test_datasets, cityscape_dataset = get_datasets(args)
    print('Done.')

    metrics = set_metrics(args)
    train_clients, test_clients = gen_clients(args, train_datasets, test_datasets, model)
    # gta_dataset = get_gta(args)
    cityscape_dataloader = DataLoader(cityscape_dataset, batch_size=args.bs, shuffle=False)
    server = Server(args, train_clients, test_clients, model, metrics)
    PATH = "/content/drive/MyDrive/MLDL23-FL-step5-fda-yolo5/checkpoints/model_step5.pt"
    # load_checkpoint = True
    # save_ckpt = False
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
                    PATH = "/content/drive/MyDrive/MLDL23-FL-step5-fda-yolo5/checkpoints/model_step5-1.pt"
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
                  PATH = "/content/drive/MyDrive/MLDL23-FL-step5-fda-yolo5/checkpoints/model_step5-1.pt"
                  print("\nloading checkpoints...")
                  opt = optim.SGD(model.parameters(), lr=args.lr, momentum=args.m,
                                      weight_decay=args.wd)
                  checkpoint = torch.load(PATH)
                  model.load_state_dict(checkpoint['model_state_dict'])
                  opt.load_state_dict(checkpoint['optimizer_state_dict'])
                  epoch = checkpoint['epoch']
                  #loss = checkpoint['loss']
                  scheduler2 = lr_scheduler.StepLR(opt, step_size=5, gamma=0.1)
                
                  model.train()
                  print("done")

    yolo_masks, yolo_prob_mask = get_yolo_prediction(yolov8_model, image_list, file_names_without_extensions)
    # yolo_masks=[]
    # yolo_prob_mask = []
    test2(args, cityscape_dataloader, metrics, model, yolo_masks, yolo_prob_mask)
    test_score = metrics['test_same_dom'].get_results()

    print(test_score)

    print("Job completed!!")


if __name__ == '__main__':
    main()