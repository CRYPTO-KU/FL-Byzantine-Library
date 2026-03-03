import math

import numpy as np
import torch
from .base import _BaseAggregator
from numpy.random import default_rng



def bucket_quotas(num_client,buck_size): ## random not filled bucket
    total_bucket = math.ceil(num_client/buck_size)
    full_buckets = num_client // buck_size
    buck_quotas = np.zeros(total_bucket)
    full_inds = np.random.choice(total_bucket,full_buckets,replace=False)
    buck_quotas[full_inds] = buck_size
    left_w = int(num_client - (full_buckets*buck_size))
    if left_w >0:
        buck_quotas[buck_quotas==0] = left_w
    return buck_quotas

def bucket_quotas2(num_client,buck_size): ## last bucket may not be filled
    total_bucket = math.ceil(num_client/buck_size)
    full_buckets = num_client // buck_size
    buck_quotas = np.zeros(total_bucket)
    buck_quotas[:full_buckets] = buck_size
    left_w = int(num_client - (full_buckets*buck_size))
    if left_w >0:
        buck_quotas[-1] = left_w
    return buck_quotas

def bucket_quotas3(num_client, buck_size): ## oversized buckets can happen
    full_buckets = num_client // buck_size
    buck_quotas = np.ones(full_buckets) * buck_size
    left_w = int(num_client - (full_buckets*buck_size))
    if left_w >0:
        np.random.choice(full_buckets,left_w,replace=False)
        buck_quotas[left_w]+=1
    return buck_quotas

def bucket_quotas4(num_client, buck_size): ## balanced
    num_buck = math.ceil(num_client/buck_size)
    buck = np.array_split(np.zeros(num_client),num_buck)
    buck_quotas = np.zeros(num_buck)
    for i,l in enumerate(buck):
        buck_quotas[i] = len(l)
    return buck_quotas

def shuffle_cluster_inds(cluster):
    new_cluster = []
    for c in cluster:
        p = torch.randperm(len(c))
        new_cluster.append(c[p])
    return cluster

class Clipping_seq(_BaseAggregator):
    def __init__(self, tau, buck_len=3,buck_avg=False,mult_ref=False):
        self.tau = tau
        self.buck_len = buck_len
        self.buck_avg = buck_avg
        super(Clipping_seq, self).__init__()
        self.momentum = None
        self.mult_ref = mult_ref
        self.cos = torch.nn.CosineSimilarity(dim=0)
        self.shuffle_clusters = False


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

    def clip(self, v):
        v_norm = torch.norm(v)
        scale = min(1, self.tau / v_norm)
        return v * scale

    def __call__(self, inputs):
        flag = 1
        if self.momentum is None:
            self.momentum = torch.zeros_like(inputs[0])
            flag = 0
        device = inputs[0].get_device()
        device = device if device > -1 else "cpu"
        if flag:
            bucket = self.bucket_cos_(inputs)
        else:
            bucket = self.buck_rand_sel(inputs)
        orig_ref = self.momentum.detach().clone()
        if self.buck_avg:
            if self.mult_ref:
                for ins in bucket.values():
                    buck_avg = sum(v.to(device) for v in ins) / len(ins)
                    self.momentum = (
                            self.clip(orig_ref + self.clip(buck_avg - orig_ref) - self.momentum)
                            + self.momentum
                    )
            else:
                for val in bucket.values():
                    self.momentum = (
                            self.clip(sum(v.to(device) for v in val) / len(val) - self.momentum)
                            + self.momentum
                    )
        else:
            if self.mult_ref:
                for ins in bucket.values():
                    self.momentum = (
                            sum(self.clip(orig_ref + self.clip(v.to(device) - orig_ref) - self.momentum)
                                for v in ins) / len(ins)
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



    # def bucket_cos_new(self,inputs):
    #     device = inputs[0].get_device()
    #     device = device if device > -1 else "cpu"
    #     buck_len = self.buck_len
    #     cluster_size = len(inputs) // buck_len
    #     sims = [self.cos(m, self.momentum) for m in inputs]
    #     sims = torch.tensor(sims, device=device)
    #     clusters = torch.split(torch.argsort(sims), cluster_size)
    #     rands = [np.random.choice(cluster_size, len(c), replace=False) for c in clusters]
    #     buckets = [[] for i in range(cluster_size)]
    #     for cluster, r in zip(clusters, rands):
    #         for buck_id, client_id in zip(r, cluster):
    #             buckets[buck_id].append(inputs[client_id])
    #     buckets = {i:bucket for i,bucket in enumerate(buckets)}
    #     return buckets
    #
    # def bucket_cos_new2(self,inputs): ## new-bug
    #     device = inputs[0].get_device()
    #     device = device if device > -1 else "cpu"
    #     num_client = len(inputs)
    #     buck_len = self.buck_len
    #     num_cluster = self.num_cluster
    #     sims = [self.cos(m, self.momentum) for m in inputs]
    #     sims = torch.tensor(sims, device=device)
    #     grads = torch.stack(inputs)[torch.argsort(sims)]
    #     clusters = torch.tensor_split(grads, num_cluster)
    #     if self.shuffle_clusters:
    #         clusters = shuffle_cluster_inds(clusters)
    #     bucket_quo = bucket_quotas4(num_client,buck_len)
    #     deb = np.zeros_like(bucket_quo)
    #     bucks = [[] for i in range(len(bucket_quo))]
    #     for i, cluster in enumerate(clusters):
    #         force_bucks = len(clusters) - i
    #         force_select = bucket_quo == force_bucks
    #         if np.sum(force_select) > 0 and force_bucks > 1:
    #             f = np.arange(len(bucket_quo))[force_select]
    #             if len(f) < len(cluster):  ## need further fix
    #                 rem = len(cluster) - len(f)
    #                 inds = (~force_select) * (bucket_quo > 0)
    #                 f_rem = np.arange(len(bucket_quo))[inds]
    #                 f_rem = np.random.choice(f_rem, rem, replace=False)
    #                 sel_b = np.concatenate((f, f_rem), axis=0)
    #             else:
    #                 sel_b = np.random.choice(f, len(cluster),replace=False)
    #         else:
    #             avail_bucks = np.arange(len(bucket_quo))[bucket_quo > 0]
    #             sel_b = np.random.choice(avail_bucks, len(cluster), replace=False)
    #         deb[sel_b] +=1
    #         [bucks[ind].append(grad) for ind, grad in zip(sel_b, cluster)]
    #         bucket_quo[sel_b] -= 1
    #     buckets = {i: bucket for i, bucket in enumerate(bucks)}
    #     return buckets
