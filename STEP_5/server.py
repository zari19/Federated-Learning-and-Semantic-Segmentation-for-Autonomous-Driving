import copy
from collections import OrderedDict
import os
import numpy as np
from PIL import Image
import torch
device = torch.device( 'cuda' if torch. cuda. is_available () else 'cpu')
from utils.stream_metrics import StreamClsMetrics, Metrics
import torchvision.transforms as transforms
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim.lr_scheduler import LambdaLR

#from main import get_dataset_num_classes

class Server:

    def __init__(self, args, train_clients, test_clients,model, metrics, optimizer=None):
        self.args = args
        self.train_clients = train_clients
        self.test_clients = test_clients
        self.model = model
        self.metrics = metrics
        self.opt_string = optimizer
        self.checkpoints_loaded_executed = False
        self.optimizer = self._get_optimizer()
        self.updates = []
        #self.get_results = StreamClsMetrics()
        self.model_params_dict = copy.deepcopy(self.model.state_dict())


    def add_updates(self, num_samples, update):
        self.updates.append((num_samples, update))

    def _get_outputs_server(self, images):
        if self.args.model == 'deeplabv3_mobilenetv2':
            return self.model(images)['out']
        if self.args.model == 'resnet18':
            return self.model(images)
        raise NotImplementedError

    def get_dataset_num_classes(self,dataset): #return dataset number of classes
        if dataset == 'idda':
            return 16
        if dataset == 'femnist':
            return 62
        raise NotImplementedError


    def load_server_model_on_client(self, client):
        client.model.load_state_dict(self.model_params_dict)

    def train_round(self, clients, metrics, current_round, opt, scheduler):
    
        """
            This method trains the model with the dataset of the clients. It handles the training at single round level
            :param clients: list of all the clients to train
            :return: model updates gathered from the clients, to be aggregated
        """
        running_loss = {}
        for i, c in enumerate(clients): 
            self.load_server_model_on_client(c) 
        
            print(f'iteration = {i}')
            print(f'client = {c}')
            
            num_train_samples, update, dict_losses_list, optimizer = c.train(metrics, current_round, opt, scheduler)
            running_loss[c] = {'loss': dict_losses_list, 'num_samples': num_train_samples
            self.add_updates(num_samples=num_train_samples, update=update)


        return running_loss, optimizer

    
    def update_model(self):
        """FedAvg on the clients' updates for the current round.
        Weighted average of self.updates, where the weight is given by the number
        of samples seen by the corresponding client at training time.
        Saves the new central model in self.client_model and its state dictionary in self.model
        """

        averaged_sol_n = self.aggregate()

        if self.optimizer is not None:  # optimizer step
            self._server_opt(averaged_sol_n)
            self.total_grad = self._get_model_total_grad()
        else:
            self.model.load_state_dict(averaged_sol_n, strict=False)
        self.model_params_dict = copy.deepcopy(self.model.state_dict())

        self.updates = []


    def _configure_optimizer(self, params, current_round):
      if self.args.optimizer == 'SGD':
          optimizer = optim.SGD(params, lr=self.args.lr, momentum=self.args.m,
                                weight_decay=self.args.wd)
      elif self.args.optimizer == 'other':
          optimizer = optim.Adam(params, lr=self.args.lr, weight_decay=self.args.wd)
    
      scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
      
      return optimizer, scheduler

    def aggregate(self):
        """
        This method handles the FedAvg aggregation
        :param updates: updates received from the clients
        :return: aggregated parameters
        """
        total_weight = 0.0 
        base = OrderedDict()
        for (client_samples, client_model) in self.updates:
          total_weight += client_samples
          for key,value in client_model.items():
            if key in base:
              base[key] += client_samples * value.type(torch.FloatTensor)
            else:
              base[key] = client_samples * value.type(torch.FloatTensor)

        averaged_update = copy.deepcopy(self.model_params_dict)
        for key, value in base.items():
          if total_weight !=0:
            averaged_update[key] = value.to('cuda')/total_weight
        return averaged_update

    def train(self, metrics, current_round):
        """
        This method orchestrates the training the evals and tests at rounds level
        """

        net = self.model
        opt, scheduler = self._configure_optimizer(net.parameters(), current_round)
        clients = self.select_clients()
        running_loss, optimizer = self.train_round(clients, metrics, current_round, opt, scheduler)
        dataset = self.args.dataset
        mtr = StreamClsMetrics(self.get_dataset_num_classes(dataset),self.get_dataset_num_classes(dataset))

        return running_loss, optimizer

    def eval_train(self):
          """
          This method handles the evaluation on the train clients
          """
          dataset = self.args.dataset
          mtr = StreamClsMetrics(self.get_dataset_num_classes(dataset),self.get_dataset_num_classes(dataset))
          print(f'num_classes = {self.get_dataset_num_classes(dataset)}')
          
          clt_images = self.select_test_clients()
          for i, c in enumerate(clt_images):
            c.test(mtr)
  
          return None

  
    def test(self,cityscape_client, metrics):

          """
              This method handles the test on the test clients
          """
          cityscape_client[0].test(metrics)

