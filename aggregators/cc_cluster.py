import math
import numpy as np
import torch
from .base import _BaseAggregator
from numpy.random import default_rng


class dummyClient:
    def __init__(self, idx,grad,malicious=False):
        self.idx = idx
        self.grad = grad
        self.malicious = malicious


class Clipping_cluster(_BaseAggregator):
    def __init__(self, tau, buck_len=3, buck_avg = True, num_clustering=3,
                 bucket_shift='random', shift_amount=1,b=5, debug_malicious=True):
        self.tau = tau
        self.buck_len = buck_len
        super(Clipping_cluster, self).__init__()
        self.momentum = None
        self.buck_avg = buck_avg
        self.cos = torch.nn.CosineSimilarity(dim=0)
        self.shuffle_clusters = False
        self.num_clustering = num_clustering
        self.bucket_shift = bucket_shift # 'sequantial' or 'random'
        self.shift_amount = shift_amount # shift amount for sequential buckets
        self.rng = np.random.default_rng()
        self.b = b


    def buck_rand_sel(self,inputs):
        device = inputs[0].get_device()
        device = device if device > -1 else "cpu"
        buck_len = self.buck_len
        cl_list = np.random.choice(len(inputs), len(inputs), replace=False)
        inputs = torch.stack(inputs)
        num_buck = len(inputs) // buck_len
        cl_list = np.array_split(cl_list, num_buck)
        buckets = []
        for cl_buck in cl_list:
            buckets.append(inputs[cl_buck])
        buckets = {i: bucket for i, bucket in enumerate(buckets)}
        return buckets
    
    def buck_rand_sel_client(self,clients):
        inputs = [c.grad for c in clients]
        bools = [c.malicious for c in clients]
        device = inputs[0].get_device()
        device = device if device > -1 else "cpu"
        buck_len = self.buck_len
        cl_list = np.random.choice(len(inputs), len(inputs), replace=False)
        inputs = torch.stack(inputs)
        num_buck = len(inputs) // buck_len
        cl_list = np.array_split(cl_list, num_buck)
        buckets = []
        buckets_bool = []
        for cl_buck in cl_list:
            buckets.append(inputs[cl_buck])
            buckets_bool.append(bools[cl_buck]) 
        buckets = {i: bucket for i, bucket in enumerate(buckets)}
        return buckets


    def bucket_cos(self, inputs): ### Non-collusion asserted
        buck_len = self.buck_len
        n = len(inputs)
        l = math.ceil(n / buck_len)
        cl_list = np.arange(n)
        bucket = {i: [] for i in range(l)}
        cos_sims = [torch.cosine_similarity(self.momentum, i, dim=0).detach().cpu().item() for i in inputs]
        cl_sorted = np.asarray(cos_sims).argsort()
        group_id = [i % l for i in cl_list]
        #device = inputs[0].get_device()
        for key in bucket.keys():
            grp = np.asarray(group_id) == key
            inds = cl_sorted[grp]
            #bucket[key] = new_inputs[cl_sorted[grp]]
            bucket[key] = [inputs[i] for i in inds]
        #for vals in bucket.values():
        #    [v.to(device) for v in vals]
        #[v.to(device) for v in inputs]
        return bucket
    
    def inputs_to_clients(self, inputs):
        dummyClients = [dummyClient(idx, grad) for idx, grad in enumerate(inputs[:-self.b])]
        dummyClients += [dummyClient(idx, grad, malicious=True) for idx, grad in enumerate(inputs[-self.b:])]
        return dummyClients


    def bucket_cos_(self,inputs):## last bucket need to fixed non-equal-buckets
        buck_len = self.buck_len
        num_client = len(inputs)
        ref = self.momentum.repeat(num_client,1)
        inputs = torch.stack(inputs)
        sims = torch.cosine_similarity(ref,inputs,dim=1)
        sort = torch.argsort(sims).long()
        #print(sort)
        inputs_sorted = inputs[sort]
        #inputs_sorted = reversed(inputs[sort]) # reversed can be used
        clusters = torch.tensor_split(inputs_sorted, buck_len)
        cls_sizes = torch.tensor([len(c) for c in clusters])
        num_bucket = math.ceil(num_client / buck_len)
        cls_perms = [torch.randperm(s) for s in cls_sizes]
        buckets = [[] for i in range(num_bucket)]
        for perm, clstr in zip(cls_perms, clusters):
            [buckets[p].append(c) for p, c in zip(perm, clstr)]
        bucket = {i: buckets[i] for i in range(len(buckets))}
        return bucket

    def bucket_cos_client(self, client):## last bucket need to fixed non-equal-buckets
        inputs = [c.grad for c in client]
        buck_len = self.buck_len
        num_client = len(inputs)
        ref = self.momentum.repeat(num_client,1)
        inputs = torch.stack(inputs)
        sims = torch.cosine_similarity(ref,inputs,dim=1)
        sort = torch.argsort(sims).long()
        #print(sort)
        is_malicious = torch.tensor([cl.malicious for cl in client],dtype=torch.bool)
        is_malicious = is_malicious[sort]
        inputs_sorted = inputs[sort]
        #inputs_sorted = reversed(inputs[sort]) # reversed can be used
        clusters = torch.tensor_split(inputs_sorted, buck_len)
        clusters_bool = torch.tensor_split(is_malicious, buck_len)
        cls_sizes = torch.tensor([len(c) for c in clusters])
        num_bucket = math.ceil(num_client / buck_len)
        cls_perms = [torch.randperm(s) for s in cls_sizes]
        buckets = [[] for i in range(num_bucket)]
        buckets_bool = [[] for i in range(num_bucket)]
        for perm, clstr, clstr_bool in zip(cls_perms, clusters, clusters_bool):
            [buckets[p].append(c) for p, c in zip(perm, clstr)]
            [buckets_bool[p].append(c) for p, c in zip(perm, clstr_bool)]
        bucket = {i: buckets[i] for i in range(len(buckets))}
        return bucket
    

    def clip(self, v):
        v_norm = torch.norm(v)
        scale = min(1, self.tau / v_norm)
        return v * scale
    
    def clip_and_update(self, grads: list, ref:torch.Tensor,avg:bool=False):
        if avg:
            mean_grad = sum(grads) / len(grads)
            grad_clipped = self.clip(mean_grad - ref)
            new_ref = ref + grad_clipped
        else:
            clipped_grads = [self.clip(g - ref) for g in grads]
            new_ref = ref + sum(clipped_grads) / len(clipped_grads)
        return new_ref
    

    def __call__(self, inputs):
        flag = 1
        if self.momentum is None:
            self.momentum = torch.zeros_like(inputs[0])
            flag = 0
        #device = inputs[0].get_device()
        #device = device if device > -1 else "cpu"
        if flag:
            bucket = self.bucket_cos_(inputs)
            for i in range(self.num_clustering):
                if i == 0:
                    refs = self.__seq_aggr__(bucket.values(), self.momentum)
                else:
                    if self.bucket_shift == 'sequential':
                        rotation = np.roll(range(len(list(bucket.values()))), self.shift_amount).tolist()
                    elif self.bucket_shift == 'random-sequantial':
                        shift_amount = self.rng.integers(0, len(refs))
                        rotation = np.roll(range(len(list(bucket.values()))), shift_amount).tolist()
                    else:
                        rotation = np.random.permutation(len(refs))
                    rotated_buckets = [list(bucket.values())[i] for i in rotation]
                    refs = self.__seq_aggr__(rotated_buckets, refs)
            self.momentum = sum(refs) / len(refs)
        else:
            buckets = self.buck_rand_sel(inputs)
            self.__buck_aggr__(buckets)
            #self.__rand_aggr__(inputs)
        
        self.momentum = self.momentum.to(inputs[0])

        return torch.clone(self.momentum).detach()
    
    def __rand_aggr__(self, inputs):
        # Randomly select a subset of inputs for aggregation
        bucket = self.buck_rand_sel(inputs)
        clipped_vals = [self.clip(sum(v for v in val) / len(val)) for val in bucket.values()]
        avg = sum(clipped_vals) / len(clipped_vals)
        self.momentum = avg + self.momentum
        return
    
    def __buck_aggr__(self, buckets):
        # Randomly select a subset of inputs for aggregation
        clipped_vals = [self.clip(sum(v for v in val) / len(val)) for val in buckets.values()]
        avg = sum(clipped_vals) / len(clipped_vals)
        self.momentum = avg + self.momentum
        return
    
    def __seq_aggr__(self, clusters,ref):
        if isinstance(ref, list):
            new_refs = [self.clip_and_update(cluster,r, avg=self.buck_avg) 
                        for cluster, r in zip(clusters, ref)]
        else:
            new_refs = [self.clip_and_update(cluster, ref, avg=self.buck_avg) 
                        for cluster in clusters]
        return new_refs



if __name__ == "__main__": 
    # Test the trimmed mean aggregator
    def test_aggr(clients=25,dim=6e5):
        # Create sample inputs
        #print(torch.sort(torch.stack(inputs,dim=0),dim=0)[0])
        inputs = [torch.randn(int(dim),device='cpu') for _ in range(clients)]
        # Initialize trimmed mean with b=1 (trim 1 from each end)
        aggr = Clipping_cluster(tau=1.0, buck_len=3)
        
        # Test aggregation
        result = aggr(inputs)
        results = aggr(inputs)
        print(f"AGGR  result: {result}")
        print(f"AGGR result2: {results}")
        

    test_aggr()
