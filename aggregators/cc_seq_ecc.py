import math

import numpy as np
import torch
from .base import _BaseAggregator
from numpy.random import default_rng



def bucket_manip(buckets,op='concat'):
    bucket_sizes = [len(b) for b in buckets]
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
    return buckets

def shuffle_cluster_inds(cluster):
    new_cluster = []
    for c in cluster:
        p = torch.randperm(len(c))
        new_cluster.append(c[p])
    return cluster

class Clipping_seq_ecc(_BaseAggregator):
    def __init__(self, args):
        self.tau = args.tau
        self.buck_len = args.buck_len
        self.buck_len_ecc = args.buck_len_ecc
        self.bucket_op = args.bucket_op
        super(Clipping_seq_ecc, self).__init__()
        self.momentum = None
        self.mult_ref = args.multi_clip
        self.fixed_ref = args.ref_fixed
        self.cos = torch.nn.CosineSimilarity(dim=0)
        self.L2 = torch.nn.PairwiseDistance(p=2)
        self.shuffle_clusters = False
        self.n_iter = args.n_iter


    def buck_rand_sel(self,inputs,ecc=False):
        buck_len = self.buck_len if not ecc else self.buck_len_ecc
        cl_list = np.random.choice(len(inputs), len(inputs), replace=False)
        inputs = torch.stack(inputs)
        num_buck = len(inputs) // buck_len
        cl_list = np.array_split(cl_list, num_buck)
        buckets = []
        for cl_buck in cl_list:
            buckets.append(inputs[cl_buck])
        buckets = {i: bucket for i, bucket in enumerate(buckets)}
        return buckets

    def bucket_cos(self, inputs, ecc=False):## last bucket need to fixed non-equal-buckets
        buck_len = self.buck_len if not ecc else self.buck_len_ecc
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
        buckets = bucket_manip(buckets,self.bucket_op)
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
            bucket = self.bucket_cos(inputs)
            ecc_bucket = self.bucket_cos(inputs)
        else:
            bucket = self.buck_rand_sel(inputs)
            ecc_bucket = self.buck_rand_sel(inputs)
        #mean = sum(v.to(device) for v in inputs) / len(inputs)
        orig_ref = self.momentum.detach().clone()
        #dists = [self.L2(v,mean) for v in inputs]
        #print(dists)
        if self.mult_ref:
            for ins in bucket.values():
                buck_avg = sum(v.to(device) for v in ins) / len(ins)
                self.momentum = (
                        self.clip(orig_ref + self.clip(buck_avg - orig_ref) - self.momentum)
                        + self.momentum
                )
            for ins in ecc_bucket.values():
                buck_avg = sum(v.to(device) for v in ins) / len(ins)
                self.momentum = (
                        sum(self.clip(orig_ref + self.clip(buck_avg - orig_ref) - self.momentum)
                            for v in ins) / len(ins)
                        + self.momentum
                )
        else:
            for val in bucket.values():
                self.momentum = (
                        self.clip(sum(v.to(device) for v in val) / len(val) - self.momentum)
                        + self.momentum
                )

            #ECC operation
            if self.fixed_ref: # fixed reference
                vals = list(ecc_bucket.values())
                bucket_avgs = [sum(v.to(device) for v in val) / len(val) for val in vals]
                for _ in range(self.n_iter):
                    self.momentum = (
                            sum(self.clip(v - self.momentum) for v in bucket_avgs) / len(bucket_avgs)
                            + self.momentum
                    )
            else: # Dynamic reference
                for val in ecc_bucket.values():
                    self.momentum = (
                            self.clip(sum(v.to(device) for v in val) / len(val) - self.momentum)
                            + self.momentum
                    )
        self.momentum = self.momentum.to(inputs[0])
        return torch.clone(self.momentum).detach()



