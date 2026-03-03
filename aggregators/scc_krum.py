import math

import numpy as np
import torch
from .base import _BaseAggregator
from numpy.random import default_rng



def shuffle_cluster_inds(cluster):
    new_cluster = []
    for c in cluster:
        p = torch.randperm(len(c))
        new_cluster.append(c[p])
    return cluster

def bucket_manip(buckets,op='concat'):
    bucket_sizes = [len(b) for b in buckets.values()]
    if 1 in bucket_sizes:
        if op == 'concat':
            buckets[-2].extend(buckets[-1])
            buckets.pop(-1)
        elif op == 'split':
            new_bucket = buckets[-2] + (buckets[-1])
            size = len(new_bucket)
            split = size // 2
            buckets[-2] = new_bucket[:split]
            buckets[-1] = new_bucket[split:]
        else:
            pass
    buckets = {i: buckets[i] for i in range(len(buckets))}
    return buckets

class Clipping_seq_krum(_BaseAggregator):
    def __init__(self, tau, m, buck_len=3, buck_avg=False,bucket_op=None):
        self.tau = tau
        self.buck_len = buck_len
        self.buck_avg = buck_avg
        self.bucket_op = bucket_op
        super(Clipping_seq_krum, self).__init__()
        self.momentum = None
        self.cos = torch.nn.CosineSimilarity(dim=0)
        self.shuffle_clusters = False
        self.m = m


    def buck_rand_sel(self,inputs):
        device = inputs[0].get_device()
        device = device if device > -1 else "cpu"
        buck_len = 3
        cl_list = np.random.choice(len(inputs), len(inputs), replace=False)
        inputs = torch.stack(inputs)
        num_buck = len(inputs) // buck_len
        cl_list = np.array_split(cl_list, num_buck)
        buckets = []
        for cl_buck in cl_list:
            buckets.append(inputs[cl_buck])
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
        device = inputs[0].get_device()
        for key in bucket.keys():
            grp = np.asarray(group_id) == key
            inds = cl_sorted[grp]
            #bucket[key] = new_inputs[cl_sorted[grp]]
            bucket[key] = [inputs[i] for i in inds]
        for vals in bucket.values():
            [v.to(device) for v in vals]
        [v.to(device) for v in inputs]
        return bucket

    def bucket_cos_(self,inputs):## last bucket need to fixed non-equal-buckets
        buck_len = self.buck_len
        num_client = len(inputs)
        inputs = torch.stack(inputs)
        sims = torch.cosine_similarity(self.momentum.unsqueeze(0), inputs, dim=1)
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

    def clip(self, v):
        v_norm = torch.norm(v)
        scale = min(1, self.tau / v_norm)
        return v * scale
    
    def multi_krum(self, inputs):
        """
        Fast Multi-Krum aggregation that selects the best updates based on pairwise distances.
        Uses vectorized operations for speed while maintaining the same logic as original Krum.
        
        Args:
            inputs: List of input tensors from clients
            
        Returns:
            List of tensors with selected updates (not aggregated)
        """
        if len(inputs) <= self.m + 2:
            # If we don't have enough inputs to apply Multi-Krum, return all
            return inputs
        
        n = len(inputs)
        f = self.m  # Number of Byzantine clients
        
        # Check if we have enough clients for Krum
        if 2 * f + 2 > n:
            return inputs
        
        # Number of clients to select
        num_krum_aggregated = n - f
        
        # Stack inputs for vectorized computation
        stacked = torch.stack(inputs, dim=0)
        
        # Compute pairwise squared distances using efficient vectorized operations
        # This computes ||x_i - x_j||^2 for all i,j pairs at once
        # Expand dimensions to get broadcasted subtraction
        expanded_a = stacked.unsqueeze(1)  # Shape: [n, 1, ...]
        expanded_b = stacked.unsqueeze(0)  # Shape: [1, n, ...]
        
        # Compute squared differences and sum
        # This is equivalent to squared Euclidean distance
        squared_diffs = (expanded_a - expanded_b) ** 2
        squared_distances = squared_diffs.sum(dim=tuple(range(2, squared_diffs.dim())))  # Sum all dimensions except first two
        
        # For each client, find its n-f-2 closest neighbors
        # Sort distances for each client
        sorted_distances, _ = torch.sort(squared_distances, dim=1)
        
        # Sum the distances to the n-f-2 closest neighbors (excluding self)
        # Skip the first value which is always 0 (distance to self)
        scores = sorted_distances[:, 1:n-f-1].sum(dim=1)
        
        # Select the clients with the smallest scores
        _, selected_indices = torch.topk(scores, k=min(num_krum_aggregated, n), largest=False)
        
        # Select the corresponding inputs
        selected_inputs = [inputs[i.item()] for i in selected_indices]
        return selected_inputs
    
    def agg_krum(self,inputs_list, n_byz):
        inputs = torch.stack(inputs_list, dim=0)  # Stack all client models
        n_cl = len(inputs)
        tensor = inputs
        squared_dists = torch.cdist(tensor, tensor).square()
        topk_dists, _ = squared_dists.topk(k=n_cl - n_byz -1, dim=-1, largest=False, sorted=False)
        scores = topk_dists.sum(dim=-1)
        _, candidate_idxs = scores.topk(k=n_cl - n_byz, dim=-1, largest=False)

        selected_inputs = [inputs_list[i.item()] for i in candidate_idxs]
        return selected_inputs

    def __call__(self, inputs):
        flag = 1
        if self.momentum is None:
            self.momentum = torch.zeros_like(inputs[0])
            flag = 0
        device = inputs[0].get_device()
        device = device if device > -1 else "cpu"
        inputs = self.multi_krum(inputs) if self.m > 0 else inputs
        #inputs = self.agg_krum(inputs, self.m) if self.m > 0 else inputs
        if flag:
            bucket = self.bucket_cos_(inputs)
        else:
            bucket = self.buck_rand_sel(inputs)
        bucket = bucket_manip(bucket,self.bucket_op)
        if self.buck_avg:
            for val in bucket.values():
                self.momentum = (
                        self.clip(sum(v.to(device) for v in val) / len(val) - self.momentum)
                        + self.momentum
                )
        else:
            for i, ins in enumerate(bucket.values()):
                self.momentum = (
                        sum(self.clip(v.to(device) - self.momentum) for v in ins) / len(ins)
                        + self.momentum
                )
        self.momentum = self.momentum.to(inputs[0])

        return torch.clone(self.momentum).detach()

