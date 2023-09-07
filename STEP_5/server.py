import copy
from collections import OrderedDict
import os
import numpy as np
from PIL import Image
import torch
#from utils import weight_train_loss
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

    def _get_optimizer(self):

          if self.opt_string is None:
              print("Running without server optimizer")
              return None

          if self.opt_string == 'SGD':
              return optim.SGD(params=self.model.parameters(), lr=self.lr, momentum=self.momentum)

          if self.opt_string == 'FedAvgm':
              return optim.SGD(params=self.model.parameters(), lr=1, momentum=0.9)

          if self.opt_string == 'Adam':
              return optim.Adam(params=self.model.parameters(), lr=self.lr, betas=(0.9, 0.99), eps=10 ** (-1))

          if self.opt_string == 'AdaGrad':
              return optim.Adagrad(params=self.model.parameters(), lr=self.lr, eps=10 ** (-2))

          raise NotImplementedError



    def select_clients(self):
        num_clients = min(self.args.clients_per_round, len(self.train_clients))
        return np.random.choice(self.train_clients, num_clients, replace=False)

    def select_test_clients(self):
        num_clients = min(self.args.clients_per_round, len(self.test_clients))
        return np.random.choice(self.test_clients, num_clients, replace=False)

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


    def weight_train_loss(self, losses):
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


    def load_server_model_on_client(self, client):
        client.model.load_state_dict(self.model_params_dict)

    def train_round(self, clients, metrics, current_round, opt, scheduler):
        # train_round_acc = 0.
        # train_round_loss = 0.
        #updates = []
        """
            This method trains the model with the dataset of the clients. It handles the training at single round level
            :param clients: list of all the clients to train
            :return: model updates gathered from the clients, to be aggregated
        """
        running_loss = {}
        for i, c in enumerate(clients): #i=#iteration from 0 to 8, c=#value of client

            self.load_server_model_on_client(c) #maybe to remove from fedavg
            # TODO: missing code here!
            #nedd to access inside the json client to get images
            print(f'iteration = {i}')
            print(f'client = {c}')
            #print(c.get_model())
            
            num_train_samples, update, dict_losses_list, optimizer = c.train(metrics, current_round, opt, scheduler)
            #print('after train')
            #self.add_updates(num_samples=num_train_samples, update=update)
            # out = self.train_round(c)
            # print('after train round')
            # num_samples, update, dict_losses_list = out
            #print(dict_losses_list)
            running_loss[c] = {'loss': dict_losses_list, 'num_samples': num_train_samples}
            #print(f'running loss = {running_loss}')
            #raise NotImplementedError
            #averaged_update = self.aggregate()#.values()
            self.add_updates(num_samples=num_train_samples, update=update)

        #print(f'update = {self.updates}')
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
      
      # Define the learning rate lambda function for LAMBDALR scheduler
      # lr_lambda = lambda epoch: 0.95 ** (current_round * self.args.num_epochs + epoch)
      # scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
      # #scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=self.polynomial_decay)
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
          #print(f'client sample = {client_samples}')
          #print(f'client model = {client_model}')
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

        # TODO: missing code here!
        #raise NotImplementedError
        return averaged_update

    def train(self, metrics, current_round):
        """
        This method orchestrates the training the evals and tests at rounds level
        """
        #train_metrics, val_metrics, ckpt_path = self.call_setup_pre_training()
        # if self.optimizer is not None:
        #     self.optimizer.zero_grad()
        net = self.model
        PATH = "/content/drive/MyDrive/MLDL23-FL-step5-fda-yolo3/checkpoints/model_step5.pt"
        if self.args.load_checkpoint== True and  (not self.checkpoints_loaded_executed):
                  print("\nloading checkpoints...")
                  opt = optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.m,
                                      weight_decay=self.args.wd)
                  checkpoint = torch.load(PATH)
                  net.load_state_dict(checkpoint['model_state_dict'])
                  opt.load_state_dict(checkpoint['optimizer_state_dict'])
                  epoch = checkpoint['epoch']
                  #loss = checkpoint['loss']
                  scheduler2 = lr_scheduler.StepLR(opt, step_size=5, gamma=0.1)
                  lr_lambda = lambda epoch: 0.95 ** (current_round * self.args.num_epochs + epoch)
                  
                  scheduler = LambdaLR(opt, lr_lambda=lr_lambda)
                  print(f'Learning rate at epoch {epoch}: {scheduler2.get_last_lr()[0]}')
                  net.train()
                  self.checkpoints_loaded_executed = True
                  print("done")
        else:
                  opt, scheduler = self._configure_optimizer(net.parameters(), current_round)
        clients = self.select_clients()
        #print(self.train_clients)

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
        # TODO: missing code here!
        #raise NotImplementedError

  
    def test2(self,cityscape_client, metrics):

          """
              This method handles the test on the test clients
          """
          # print(type(test_clients[0]))
          # print(test_clients[0].test_loader.dataset.list_samples[0:5])
          cityscape_client[0].test2(metrics)

