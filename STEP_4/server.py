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

#from main import get_dataset_num_classes

class Server:

    def __init__(self, args, train_clients, test_clients, model, metrics, optimizer=None):
        self.args = args
        self.train_clients = train_clients
        self.test_clients = test_clients
        self.model = model
        self.metrics = metrics
        self.opt_string = optimizer
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

    def train_round(self, clients, metrics, student_model, teacher_model):
        """
            This method trains the model with the dataset of the clients. It handles the training at single round level
            :param clients: list of all the clients to train
            :return: model updates gathered from the clients, to be aggregated
        """
        running_loss = {}
        for i, c in enumerate(clients):

            print(f'iteration = {i}')
            print(f'client = {c}')
            
            num_train_samples, update, dict_losses_list = c.train(metrics, student_model, teacher_model)

            running_loss[c] = {'loss': dict_losses_list, 'num_samples': num_train_samples}
            #print(f'running loss = {running_loss}')
            #raise NotImplementedError
            #averaged_update = self.aggregate()
            self.add_updates(num_samples=num_train_samples, update=update)

        #print(f'update = {self.updates}')
        return running_loss

    
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

    def train(self, metrics, student_model, teacher_model):
        """
        This method orchestrates the training the evals and tests at rounds level
        """
        if self.optimizer is not None:
            self.optimizer.zero_grad()
        clients = self.select_clients()

        running_loss = self.train_round(clients, metrics, student_model, teacher_model)
        return running_loss

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

  
    def test2(self,test_clients, metrics, model):

          """
                    This method handles the test on the test clients
          """
          test_clients[1].test2(metrics)
