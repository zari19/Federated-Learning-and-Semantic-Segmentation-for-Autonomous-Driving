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
from torch.optim.lr_scheduler import LambdaLR


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
        metrics.update(labels, prediction)

    def _get_outputs(self, images):
        if self.args.model == 'deeplabv3_mobilenetv2':
            return self.model(images)['out']
        if self.args.model == 'resnet18':
            return self.model(images)
        raise NotImplementedError

    def get_optimizer(self, net, lr, wd, momentum):
      optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=wd, momentum=momentum)
      return optimizer

    def loss_function(self):
      loss_function = nn.CrossEntropyLoss()
      return loss_function

    def calc_losses(self, images, labels):
      if self.args.model == 'deeplabv3_mobilenetv2':
          outputs = self._get_outputs(images)
          loss_tot = self.reduction(self.criterion(outputs, labels), labels)
          dict_calc_losses = {'loss_tot': loss_tot}
      else:
          raise NotImplementedError

      return dict_calc_losses, outputs

    def calc_loss_fed(dict_losses):
        return dict_losses
      
    def generate_update(self):
        return copy.deepcopy(self.model.state_dict())


    def _configure_optimizer(self, params, current_round):
  
      optimizer = optim.SGD(params, lr=self.args.lr, momentum=0.9,
                                weight_decay=self.args.wd)
      
    
      lr_lambda = lambda epoch: 0.95 ** (current_round * self.args.num_epochs + epoch)
      scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
      
      return optimizer, scheduler

    def handle_log_loss(self, dict_all_epoch_losses, dict_losses_list):

        for n, l in dict_all_epoch_losses.items():

            dict_all_epoch_losses[n] = torch.tensor(l).to(device)

        return dict_all_epoch_losses, dict_losses_list


    def run_epoch(self, cur_epoch, optimizer, metrics, scheduler=None):
        """
        This method locally trains the model with the dataset of the client. It handles the training at mini-batch level
        :param cur_epoch: current epoch of training
        :param optimizer: optimizer used for the local training
        """
        dict_all_epoch_losses = defaultdict(lambda: 0)

        for cur_step, (images, labels) in enumerate(self.train_loader):
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)
            optimizer.zero_grad()
            dict_calc_losses, outputs = self.calc_losses(images, labels)
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

        return dict_all_epoch_losses
            
        

    def train(self, metrics):
        """
        This method locally trains the model with the dataset of the client. It handles the training at epochs level
        (by calling the run_epoch method for each local epoch of training)
        :return: length of the local dataset, copy of the model parameters
        """
        num_train_samples = len(self.dataset)
        dict_losses_list = defaultdict(lambda: [])
        self.model.train()
        net = self.get_model()
        
        for epoch in range(self.args.num_epochs):
        
            opt, scheduler = self._configure_optimizer(net.parameters(), epoch)
            dict_all_epoch_losses = self.run_epoch(epoch, optimizer = opt, metrics=metrics, scheduler=scheduler)
            dict_all_epoch_losses, dict_losses_list = self.handle_log_loss(dict_all_epoch_losses, dict_losses_list)

        update = self.generate_update()

        return num_train_samples, update, dict_losses_list

    def test2(self, metric):
        """
        This method tests the model on the local dataset of the client.
        :param metric: StreamMetric object
        """
        print("Test...")
        self.model.eval()
        class_loss = 0.0
        ret_samples = []

        with torch.no_grad():
            for i, sample in enumerate(self.test_loader):
              images, labels = sample
              
              images = images.to(device, dtype=torch.float32)
              labels = labels.to(device, dtype=torch.long)
              outputs = self._get_outputs(images)

              loss = self.reduction(self.criterion(outputs, labels),labels)
              class_loss += loss.item()

              _, prediction = outputs.max(dim=1)
              labels = labels.cpu().numpy()
              prediction = prediction.cpu().numpy()
              metric['test_same_dom'].update(labels, prediction)


              if self.args.plot == True:
                  pred2 = prediction[0,:,:] 
                  plt.imshow(pred2)
                  plt.savefig('root/pred{}.png'.format(i))

            class_loss = torch.tensor(class_loss).to(device)
            print(f'class_loss = {class_loss}')
            class_loss = class_loss / len(self.test_loader)

        return class_loss, ret_samples
