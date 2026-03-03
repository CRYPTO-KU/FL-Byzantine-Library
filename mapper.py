from data_loader import DatasetSplit , get_dataset,get_indices, seperate_root_dataset
from model_registry import get_net
from aggregators.aggr_mapper import get_aggr, get_bucketing
#from pruners.prune_mapper import get_attack_locations
from pruners.prune_mapper import get_attack_locations
from attacks.attack_mapper import get_attacker_client
from client import client
from fl import FL 
from utils import count_parameters, get_layer_dims_lasa
import numpy as np

class Mapper:
    def __init__(self, args, device):
        self.args = args
        self.num_client = args.num_client
        self.participation = args.cl_part
        self.b = int(args.traitor*self.num_client) if args.traitor < 1 else int(args.traitor)
        self.sparse_attack = True if args.attack.split('_')[0] == 'sparse' else False
        self.train_set = None
        self.net_ps = None
        self.device = device
        
    def get_benign_cl(self,id,dataset,device,args):
        client_params = {'id': id, 'dataset': dataset, 'device': device, 'args': args}
        return client(**client_params)

    def get_attacker_cl(self,id,dataset,device,args):
        return get_attacker_client(id,dataset,device,args)
        

    def get_robust_aggregator(self,net_ps,device,num_client,root_dataset=None,lassa_dims=None):
        num_traitors= int(num_client * self.args.traitor) if self.args.traitor < 1 else int(self.args.traitor)
        if self.args.bucketing:
            num_psuedo_clients = int(self.num_client * self.participation) // self.args.buck_len
            if self.num_client % self.args.buck_len != 0 and self.args.bucket_op != 'concat':
                num_psuedo_clients += 1
            psuedo_traitors = num_traitors // self.args.buck_len        
        else:
            num_psuedo_clients = int(self.num_client * self.participation)
            psuedo_traitors = int(num_traitors * self.participation)
        aggregator = get_aggr(self.args, net_ps, device, num_psuedo_clients, psuedo_traitors,  root_dataset=root_dataset, lasa_dims=lassa_dims)
        bucket_fn = get_bucketing(self.args)
        return aggregator, bucket_fn

    def initialize_FL(self) -> FL:
        # get dataset and network
        train_set, test_set = get_dataset(self.args)
        net_ps = get_net(self.args).to(self.device)
        print('number of parameters ', round(count_parameters(net_ps) / 1e6,3) ,'M')
        root_dataset,lasa_layers = None, None
        if self.args.aggr in ('fl_trust', 'skymask'):
            train_set, root_dataset = seperate_root_dataset(train_set, 50)
        lasa_layers = get_layer_dims_lasa(net_ps)
        client_inds, data_map = get_indices(train_set,self.args)
        total_samples = len(train_set)
        benign_inds, malicious_inds = [],[]
        # Initialize benign and malicious clients
        benign_clients = []
        malicious_clients = []
        if self.b > 0:
            traitors = np.random.choice(range(self.num_client), self.b, replace=False)
        else:
            traitors = []
        for i in range(self.num_client):
            worker_dataset = DatasetSplit(train_set, client_inds[i])
            # worker_data_map = data_map[i]
            if i in traitors:
                malicious_clients.append(get_attacker_client(i, worker_dataset, self.device, self.args,layer_inds=lasa_layers))
                malicious_inds.append(client_inds[i])
                if len(malicious_clients) == 1:
                    malicious_clients[0].debug_byz = True
            else:
                benign_clients.append(self.get_benign_cl(i, worker_dataset, self.device, self.args))
                benign_inds.append(client_inds[i])

        # Initialize pruning for sparse location masks
        if self.sparse_attack:
            prune_mask, bn_mask = get_attack_locations(self.args, net_ps, train_set, malicious_inds, benign_inds, self.device)
            # Benign clients can also have a mask to reduce communication, currently not used
            for cl in malicious_clients:
                cl.update_mask(prune_mask)

        # Initialize aggregator and bucket function
        aggregator, bucket_fn = self.get_robust_aggregator(net_ps, self.device, self.num_client, root_dataset=root_dataset,lassa_dims=lasa_layers)
        # Set Federated Learning parameters
        fl_params = {
            'args': self.args,
            'total_samples': total_samples,
            'net_ps': net_ps,
            'test_set': test_set,
            'benign_clients': benign_clients,
            'malicious_clients': malicious_clients,
            'aggr': aggregator,
            'bucket': bucket_fn,
            'device': self.device
        }
        fl = FL(**fl_params)
        return fl