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
        #print(len(selected_benigns),len(selected_malicious))
        #training
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


    def std_analysis(self,all_grads):
        stacked_gradients = torch.stack(all_grads, 1)
        mu = torch.mean(stacked_gradients, 1).to(self.device)
        std = torch.std(stacked_gradients, 1).to(self.device)
        stacked_gradients = stacked_gradients.permute(1,0)
        distance_to_mean = torch.abs((stacked_gradients - mu) / std)
        std_vec = distance_to_mean
        outliers_val,outliers_inds = torch.topk(std_vec,dim=0, k=5, largest=True)
        non_outlier_val, _ = torch.topk(std_vec,dim=0, k=20, largest=False)
        all_median,_ = torch.median(std_vec, dim=0)
        max_non_outlier,_ = torch.max(non_outlier_val,0)
        average_outlier_distance = torch.mean(outliers_val, 0)
        outlier_median, _ = torch.median(outliers_val, dim=0)
        cheby_density = (std_vec > 1.41).float().mean(0)
        cheby_density2 = (non_outlier_val > 1.41).float().sum(0)
        locs,crits = get_layer_dims2(self.net_ps)
        layer_names = get_layer_names(self.net_ps)
        if self.layer_inspection is None:
            self.create_layer_dict()
        for inds, is_crit,name in zip(locs,crits,layer_names):
            if not is_crit: # weights only
                self.layer_inspection['median'][name].append(all_median[inds].mean().detach().cpu().numpy())
                self.layer_inspection['avg_outlier_dist'][name].append(average_outlier_distance[inds].mean().detach().cpu().numpy())
                self.layer_inspection['outlier_median'][name].append(outlier_median[inds].mean().detach().cpu().numpy())
                self.layer_inspection['density'][name].append(cheby_density[inds].mean().detach().cpu().numpy())
                self.layer_inspection['density2'][name].append(cheby_density2[inds].mean().detach().cpu().numpy())


    def layer_std_analysis(self,epoch):
        layers = get_layer_names(self.net_ps)
        locs, crits = get_layer_dims2(self.net_ps)
        selected_layers = [name for name,crit in zip(layers,crits) if not crit]

        x = np.arange(len(selected_layers))  # the label locations
        width = 0.2  # the width of the bars
        multiplier = 0

        fig, ax = plt.subplots(layout='constrained')

        layer_dic = self.layer_inspection
        final_dic = {key:[] for key in self.layer_dic.keys()}
        for key in final_dic.keys():
            for layer in layer_dic[key]:
                final_dic[key].append(np.mean(layer_dic[key][layer]))

        for attribute, measurement in final_dic.items():
            offset = width * multiplier
            rects = ax.bar(x + offset, np.round(measurement,2), width, label=attribute)
            ax.bar_label(rects, padding=3)
            multiplier += 1

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('STD multipler')
        ax.set_xlabel('Layers')
        ax.set_xticks(x + width, selected_layers)
        ax.legend(loc='upper left')
        fig = plt.gcf()
        fig.set_size_inches(26, 10)
        plt.savefig('STDFigs/layer_std_analysis{}-{}-{}.png'.format(self.args.aggr,self.args.dataset_dist,epoch))
        #plt.show()
        return

    def create_layer_dict(self):
        locs,crits = get_layer_dims2(self.net_ps)
        layer_names = get_layer_names(self.net_ps)

        layer_dic = {name:[] for name,crit in zip(layer_names,crits) if not crit}
        final_dic = {key:deepcopy(layer_dic) for key in self.layer_dic.keys()}
        self.layer_inspection = final_dic