"""
Not an aggregator, but a variance reduction method by clustering the clients
before aggregation.
"""
import torch
import numpy as np
import math

class Bucketing(object):
    def __init__(self, buck_len, bucketing_type='Random',bucket_op=None):
        self.buck_len = buck_len
        self.buck_type = bucketing_type
        self.bucket_op = bucket_op
        self.L2 = torch.nn.PairwiseDistance(p=2) 

    def bucket_clients(self,inputs,ref=None):
        if self.buck_type == 'Random' or ref is None:
            bucket = self.buck_rand_sel(inputs,ref)
        elif self.buck_type == 'Cosine Distance':
            bucket = self.bucket_cosine_distance(inputs,ref)
        elif self.buck_type == 'L2 distance':
            bucket = self.bucket_L2(inputs,ref)
        else:
            raise ValueError('Unknown bucketing type: {}'.format(self.buck_type))
        
        self.num_psuedo_clients = len(bucket)
        reduced_inputs = [sum(vals) / len(vals) for vals in bucket]
        return reduced_inputs
    
    def __call__(self, inputs,ref):
        return self.bucket_clients(inputs,ref)
    

    def buck_rand_sel(self,inputs,ref=None):
        buck_len = self.buck_len
        cl_list = np.random.choice(len(inputs), len(inputs), replace=False)
        inputs = torch.stack(inputs)
        num_buck = len(inputs) // buck_len
        cl_list = np.array_split(cl_list, num_buck)
        buckets = []
        for cl_buck in cl_list:
            buckets.append(inputs[cl_buck])
        buckets = self.bucket_manip(buckets, self.bucket_op)
        return buckets
    
    def bucket_cosine_distance(self,inputs,reference):## last bucket need to fixed non-equal-buckets
        buck_len = self.buck_len
        num_client = len(inputs)
        ref = reference.repeat(num_client,1)
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
        buckets = self.bucket_manip(buckets, self.bucket_op)
        return buckets
    
    def bucket_L2_normed(self,inputs,reference):## last bucket need to fixed non-equal-buckets
        buck_len = self.buck_len
        num_client = len(inputs)
        mean = sum(v for v in inputs) / len(inputs)
        dists = [self.L2(v, mean) for v in inputs]
        norms = [torch.norm(v).item() for v in inputs]
        norm_scale = [max(1,n) for n in norms]
        sort = torch.argsort(torch.tensor(dists)).long()
        normed_inputs = [v / norm_scale[i] for i, v in enumerate(inputs)]
        inputs_sorted = torch.stack(normed_inputs)[sort]
        # inputs_sorted = reversed(inputs[sort]) # reversed can be used
        clusters = torch.tensor_split(inputs_sorted, buck_len)
        cls_sizes = torch.tensor([len(c) for c in clusters])
        num_bucket = math.ceil(num_client / buck_len)
        cls_perms = [torch.randperm(s) for s in cls_sizes]
        buckets = [[] for i in range(num_bucket)]
        for perm, clstr in zip(cls_perms, clusters):
            [buckets[p].append(c) for p, c in zip(perm, clstr)]
        buckets = self.bucket_manip(buckets, self.bucket_op)
        return buckets
    
    def bucket_L2(self,inputs,reference):## last bucket need to fixed non-equal-buckets
        buck_len = self.buck_len
        num_client = len(inputs)
        dists = [self.L2(v, reference) for v in inputs]
        sort = torch.argsort(torch.tensor(dists)).long()
        inputs_sorted = torch.stack(inputs)[sort]
        # inputs_sorted = reversed(inputs[sort]) # reversed can be used
        clusters = torch.tensor_split(inputs_sorted, buck_len)
        cls_sizes = torch.tensor([len(c) for c in clusters])
        num_bucket = math.ceil(num_client / buck_len)
        cls_perms = [torch.randperm(s) for s in cls_sizes]
        buckets = [[] for i in range(num_bucket)]
        for perm, clstr in zip(cls_perms, clusters):
            [buckets[p].append(c) for p, c in zip(perm, clstr)]
        buckets = self.bucket_manip(buckets, self.bucket_op)
        return buckets

    def bucket_manip(self,buckets,op=None):
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
            elif op == None:
                pass
        return buckets
