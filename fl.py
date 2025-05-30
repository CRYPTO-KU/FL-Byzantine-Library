import torch
import torch.nn as nn

from utils import count_parameters,get_grad_flattened,get_layer_dims2,get_layer_names
import data_loader as dl
import numpy as np
from mapper import *
from copy import deepcopy
import matplotlib.pyplot as plt

class FL(object):
    def __init__(self, args,train_set,client_inds, net_ps,aggr, device,prune_mask=None,bn_mask=None):
        self.args = args
        self.train_set = train_set
        self.sample_inds = client_inds
        self.net_ps = net_ps
        self.num_client = args.num_client
        self.num_malicous = int(self.num_client * args.traitor) if args.traitor < 1 else args.traitor
        self.device = device
        self.benign_clients = []
        self.malicious_clients = []
        self.aggr = aggr
        self.mask = prune_mask
        self.bn_mask = bn_mask
        self.cli_rate = args.cl_part
        self.setup_fl()
        self.layer_std_stads = None
        self.epoch = 0
        self.comm_rounds = 0
        self.avg_train_loss = 0
        self.psuedo_momentums = None
        self.layer_dic = {'median':None,'avg_outlier_dist': None,
                     'outlier_median': None, 'density': None, 'density2': None}
        self.layer_inspection = None

    def setup_fl(self):
        self.benign_clients = []
        if self.num_malicous > 0:
            traitors = np.random.choice(range(self.num_client), self.num_malicous, replace=False)
        else:
            traitors = []
        self.traitors = traitors
        for i in range(self.num_client):
            worker_dataset = dl.DatasetSplit(self.train_set, self.sample_inds[i])
            # worker_data_map = data_map[i]
            if i in traitors:
                self.malicious_clients.append(get_attacker_cl(i, worker_dataset, self.device, self.args, self.mask,self.bn_mask))
                if len(self.malicious_clients) == 1:
                    self.malicious_clients[0].debug_byz = True
            else:
                self.benign_clients.append(get_benign_cl(i, worker_dataset, self.device, self.args))

    def cross_silo_step(self):
        [cl.update_model(self.net_ps) for cl in [*self.benign_clients,*self.malicious_clients]]
        [client.train_() for client in self.benign_clients]
        benign_grads = [cl.get_grad() for cl in self.benign_clients]
        if self.args.MITM:
            byzantine_preds = benign_grads
        else:
            self.Byzantine_grad_preds()
            byzantine_preds = self.psuedo_momentums
        if len(self.malicious_clients) > 0:
            if self.malicious_clients[0].omniscient:
                self.malicious_clients[0].omniscient_callback(byzantine_preds)
                traitor_grads = [self.malicious_clients[0].adv_momentum for cl in
                                 self.malicious_clients]  # faster to do this than to call the function again
            else:
                [client.train_() for client in self.malicious_clients]
                traitor_grads = [cl.get_grad() for cl in self.malicious_clients]
            all_grads = benign_grads + traitor_grads
        else:
            all_grads = benign_grads
        self.avg_train_loss = sum([cl.mean_loss for cl in self.benign_clients]) / len(self.benign_clients)
        self.epoch += (len(all_grads) * self.args.localIter * self.benign_clients[0].bs) / len(self.train_set)
        return all_grads

    def cross_device_step(self):
        all_clients = self.benign_clients + self.malicious_clients
        all_clients = np.random.choice(all_clients, int(self.cli_rate * len(all_clients)), replace=False)
        #Client selection
        selected_benigns = [client for client in all_clients if client.id not in self.traitors]
        selected_malicious = [client for client in all_clients if client.id in self.traitors]
        [client.train_(net=self.net_ps) for client in selected_benigns]
        benign_grads = [cl.get_grad() for cl in selected_benigns]
        if len(selected_malicious) > 0:
            if selected_malicious[0].omniscient:
                selected_malicious[0].omniscient_callback(benign_grads)
                traitor_grads = [selected_malicious[0].adv_momentum for cl in
                              selected_malicious]  # faster to do this than to call the function again
            else:
                [client.train_(net=self.net_ps) for client in selected_malicious]
                traitor_grads = [cl.get_grad() for cl in selected_malicious]
            all_grads = benign_grads + traitor_grads
        else:
            all_grads = benign_grads
        self.avg_train_loss = sum([cl.mean_loss for cl in selected_benigns]) / len(selected_benigns)
        self.epoch += (len(all_grads)* self.args.localIter * selected_benigns[0].bs) / len(self.train_set)
        return all_grads

    def __get_current_epoch__(self):
        return self.epoch

    def __update_lr__(self):
        [cl.lr_step()for cl in [*self.benign_clients,*self.malicious_clients]]

    def train(self):
        self.comm_rounds +=1
        if self.args.cl_part < 1:
            all_grads = self.cross_device_step()
        else:
            all_grads = self.cross_silo_step()
        #self.std_analysis(all_grads)
        return all_grads

    def Byzantine_grad_preds(self):
        '''
        Function for non-omniscient Byzantine Attackers.
        Attackers use their own data to generate Psuedo-Grads and momentums
        '''
        pred_per_byzantine = int((1-self.args.traitor) / self.args.traitor)
        if self.psuedo_momentums is None:
            self.psuedo_momentums = [torch.tensor(torch.zeros(count_parameters(self.net_ps)),device=self.device) for cl in self.benign_clients]
        net = deepcopy(self.net_ps).to(self.device)
        criterion = nn.CrossEntropyLoss()
        c = 0
        for cl in self.malicious_clients:
            for i,data in enumerate(cl.loader):
                net.zero_grad()
                if i == pred_per_byzantine:
                    break
                x, y = data
                x, y = x.to(self.device), y.to(self.device)
                logits = net(x)
                loss = criterion(logits, y)
                loss.backward()
                grad = get_grad_flattened(net,device=self.device)
                self.psuedo_momentums[c] = self.psuedo_momentums[c] * self.args.Lmomentum + grad * (1 - self.args.Lmomentum)
                c +=1


    def aggregate(self, all_grads):
        return self.aggr.__call__(all_grads)


