import copy
import torch
import torch.nn.functional as F
import os
import numpy as np
from PIL import Image
from torch import optim, nn
from collections import defaultdict
from torch.utils.data import DataLoader
from utils.utils import get_scheduler
import matplotlib.pyplot as plt
from utils.utils import HardNegativeMining, MeanReduction
from torch import distributed
import torchvision.transforms
from utils.dist_utils import initialize_distributed, setup, find_free_port
import torch.optim.lr_scheduler as lr_scheduler

device = torch.device( 'cuda' if torch. cuda. is_available () else 'cpu')

class Client:

    def __init__(self, args, dataset, model, test_client=False):
        self.args = args
        self.dataset = dataset
        self.name = self.dataset.client_name
        self.model = model
        self.train_loader = DataLoader(self.dataset, batch_size=self.args.bs, shuffle=True, drop_last=True) \
            if not test_client else None
        self.test_loader = DataLoader(self.dataset, batch_size=self.args.bs, shuffle=False)
        self.criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='none')
        self.reduction = HardNegativeMining() if self.args.hnm else MeanReduction()
        
    def __str__(self):
        return self.name


    def get_model(self):
        return self.model


    @staticmethod
    def update_metric(metrics, outputs, labels, cur_step):
        _, prediction = outputs.max(dim=1)
        labels = labels.cpu().numpy()
        prediction = prediction.cpu().numpy()
        # print(f'update_metric_prediction_type = {type(prediction)}')
        # print(f'update_metric_prediction_shape = {prediction.shape}')

        # pred = prediction[0,:,:]
        # plt.imshow(pred)
        # plt.savefig('trial_imgs/pred{}.png'.format(cur_step))

        metrics.update(labels, prediction)

    def _get_outputs(self, images):
        if self.args.model == 'deeplabv3_mobilenetv2':
            return self.model(images)['out']
        if self.args.model == 'resnet18':
            return self.model(images)
        raise NotImplementedError

    def get_dataset_num_classes(self, dataset): #return dataset number of classes
        if dataset == 'idda':
            return 16
        if dataset == 'femnist':
            return 62
        raise NotImplementedError

    def get_optimizer(self, net, lr, wd, momentum):
      optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=wd, momentum=momentum)
      return optimizer

    def loss_function(self):
      loss_function = nn.CrossEntropyLoss()
      return loss_function

    def calculate_class_weights(self, labels):
        class_weights = torch.zeros(torch.max(labels) + 1)

        # Count the frequency of each class
        unique, counts = torch.unique(labels, return_counts=True)
        class_frequency = dict(zip(unique.cpu().numpy(), counts.cpu().numpy()))

        # Calculate class weights using inverse frequency
        total_samples = torch.sum(torch.tensor(list(class_frequency.values())))
        for class_label, frequency in class_frequency.items():
            class_weights[class_label] = total_samples / (frequency * len(class_frequency))

        #class_weights = class_weights.tolist()
        #print(class_weights)
        class_weights = torch.cat((class_weights[:15], class_weights[-1:]))
        #print(class_weights)
        return class_weights


    def calc_losses(self, images, labels):
      if self.args.model == 'deeplabv3_mobilenetv2':

          outputs = self._get_outputs(images)
          # print(type(outputs))
          # print(outputs.shape)
          # print(labels.shape)
          #labels = F.interpolate(labels, size=(540, 960), mode='bilinear', align_corners=False)

          # if outputs.size != (1920, 1080):
          #       outputs = F.interpolate(outputs, size=(540, 960), mode='bilinear', align_corners=False)


          w = self.calculate_class_weights(labels)
          #new_array = original_array[0:15] + [original_array[-1]]
          # class_weights_dict = {i: w for i, w in enumerate(w)}
          # w = torch.tensor(list(class_weights_dict.values()))
          # print(w)
          w = w.to(device, dtype=torch.float32)
          # print(w)
          # print(len(w))
          #criterion = nn.CrossEntropyLoss(w)
          criterion = nn.CrossEntropyLoss(ignore_index=255, weight = w, reduction='none')

          #print(labels)
          loss_tot = self.reduction(criterion(outputs, labels), labels)
          dict_calc_losses = {'loss_tot': loss_tot}
      else:
          raise NotImplementedError

      return dict_calc_losses, outputs
      
    def handle_grad(self, loss_tot):
        pass

    def calc_loss_fed(dict_losses):
        return dict_losses
      
    def clip_grad(self):
        pass

    def generate_update(self):
        return copy.deepcopy(self.model.state_dict())


    def _configure_optimizer(self, params):
          if self.args.optimizer == 'SGD':
              optimizer = optim.SGD(params, lr=self.args.lr, momentum=self.args.momentum,
                                    weight_decay=self.args.weight_decay, nesterov=self.args.nesterov)
          else:
              optimizer = optim.Adam(params, lr=self.args.lr, weight_decay=self.args.weight_decay)
          scheduler = get_scheduler(self.args, optimizer,
                                    max_iter=10000 * self.args.num_epochs * len(self.loader))
          return optimizer, scheduler

    def handle_log_loss(self, dict_all_epoch_losses, dict_losses_list):

        for n, l in dict_all_epoch_losses.items():

            dict_all_epoch_losses[n] = torch.tensor(l).to(device)
            #dict_losses_list[n].append(dict_all_epoch_losses[n])
        return dict_all_epoch_losses, dict_losses_list


    def run_epoch(self, cur_epoch, optimizer, metrics, scheduler=None):
        """
        This method locally trains the model with the dataset of the client. It handles the training at mini-batch level
        :param cur_epoch: current epoch of training
        :param optimizer: optimizer used for the local training
        """
        dict_all_epoch_losses = defaultdict(lambda: 0)

        for cur_step, (images, labels) in enumerate(self.train_loader):
            # TODO: missing code here!
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)
            # print("client transform")
            # print(np.unique(labels.cpu().numpy()))
            # print(f'train_images_shape = {images.shape}')
            # print(f'train_output_shape = {labels.shape}')

            optimizer.zero_grad()
            dict_calc_losses, outputs = self.calc_losses(images, labels)
            #print(f'outputs after calculate_loss = {outputs.shape}')

            dict_calc_losses['loss_tot'].backward()
            self.handle_grad(dict_calc_losses['loss_tot'])

            self.clip_grad()
            optimizer.step()
            scheduler.step()

            if cur_epoch == self.args.num_epochs - 1:
              
              self.update_metric(metrics, outputs, labels, cur_step)

            print_string = ""
            for name, l in dict_calc_losses.items():
                  if type(l) != int:
                      dict_all_epoch_losses[name] += l.detach().item()
                  else:
                      dict_all_epoch_losses[name] += l


        for name, l in dict_all_epoch_losses.items():
          dict_all_epoch_losses[name] /= len(self.train_loader)
          print_string += f"{name}={'%.3f' % dict_all_epoch_losses[name]}, "
          print(print_string)

        if self.args.ckpt:
            checkpoint = {
            'epoch': cur_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': dict_calc_losses['loss_tot'],
            }
            torch.save(checkpoint, 'checkpoint.pt')

        return dict_all_epoch_losses
            
        

    def train(self, metrics):
        """
        This method locally trains the model with the dataset of the client. It handles the training at epochs level
        (by calling the run_epoch method for each local epoch of training)
        :return: length of the local dataset, copy of the model parameters
        """
        num_train_samples = len(self.dataset)
        #optimizer, scheduler = self._configure_optimizer(params)
        dict_losses_list = defaultdict(lambda: [])
        self.model.train()
        #bn_dict_tmp = None
        net = self.get_model()
        opt = self.get_optimizer(net, lr=self.args.lr, wd=self.args.wd, momentum=self.args.m)
        scheduler = lr_scheduler.StepLR(opt, step_size=5, gamma=0.1)

        # if self.args.ckpt:
        #     checkpoint = torch.load(checkpoint.pt)
        #     self.model.load_state_dict(checkpoint['model_state_dict'])
        #     opt.load_state_dict(checkpoint['optimizer_state_dict'])
        #     epoch = checkpoint['epoch']
        #     loss = checkpoint['loss']

        #self.model.train()
        # TODO: missing code here!
        for epoch in range(self.args.num_epochs):
            # TODO: missing code here!
            
            dict_all_epoch_losses = self.run_epoch(epoch, optimizer = opt, metrics=metrics, scheduler=scheduler)
            dict_all_epoch_losses, dict_losses_list = self.handle_log_loss(dict_all_epoch_losses, dict_losses_list)

        #metrics.synch(self.device)

        update = self.generate_update()

        return num_train_samples, update, dict_losses_list

    def subs_bn_stats(self, domain, train_cl_bn_stats):
        pass

    def copy_bn_stats(self):
        pass


    def test2(self, metric):
        """
        This method tests the model on the local dataset of the client.
        :param metric: StreamMetric object
        """
        print("sambuconeCuravito")
        self.model.eval()
        class_loss = 0.0
        ret_samples = []

        # if loader is None:
        #     loader = self.loader

        with torch.no_grad():
            for i, sample in enumerate(self.test_loader):
              print(f'loading image = {self.test_loader.dataset.list_samples[i]}')
              images, labels = sample
              
              images = images.to(device, dtype=torch.float32)
              labels = labels.to(device, dtype=torch.long)
              # print(f'test_images_shape = {images.shape}')
              # print(f'test_output_shape = {labels.shape}')
              outputs = self._get_outputs(images)
              # print(f'test_output_shape = {outputs.shape}')
              # print(f'test_output_shape = {labels.shape}')

              # im = labels.cpu().numpy()
              # im = im.transpose(0, 1, 2)
              # plt.imshow(im[0])
              # plt.axis('off')
              # plt.savefig("idda_label_test{}".format(i))

              # im = images.cpu().numpy()
              # im = im.transpose(2, 3, 1)
              # plt.imshow(im[0])
              # plt.axis('off')
              # plt.savefig("idda_test{}".format(i))

              loss = self.reduction(self.criterion(outputs, labels),labels)
              class_loss += loss.item()

              _, prediction = outputs.max(dim=1)
              labels = labels.cpu().numpy()
              prediction = prediction.cpu().numpy()
              print(f'prediction shape {i} = {prediction.shape}')
              print(f'pred_test = {np.unique(prediction)}')
              unique_values, counts = np.unique(prediction, return_counts=True)

              # Print the counts for each unique value
              # for value, count in zip(unique_values, counts):
              #     print("Value:", value, "Count:", count)
              metric['test_same_dom'].update(labels, prediction)


              if self.args.plot == True:
                  pred2 = prediction[0,:,:]  # Select the first image from the batch
                  plt.imshow(pred2)
                  plt.savefig(test_root + '/pred{}.png'.format(i))

            class_loss = torch.tensor(class_loss).to(device)
            print(f'class_loss = {class_loss}')
            class_loss = class_loss / len(self.test_loader)

        return class_loss, ret_samples

      

