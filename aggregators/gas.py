from .base import _BaseAggregator
import torch
import numpy as np

def _compute_euclidean_distance(v1, v2):
    return (v1 - v2).norm()


def pairwise_euclidean_distances(vectors):
    """Compute the pairwise euclidean distance.
    Arguments:
        vectors {list} -- A list of vectors.
    Returns:
        dict -- A dict of dict of distances {i:{j:distance}}
    """
    n = len(vectors)

    distances = {}
    for i in range(n - 1):
        distances[i] = {}
        for j in range(i + 1, n):
            distances[i][j] = _compute_euclidean_distance(vectors[i], vectors[j]) ** 2
    return distances

def TM(inputs,b):
    if len(inputs) - 2 * b > 0:
        b = b
    else:
        b = b
        while len(inputs) - 2 * b <= 0:
            b -= 1
        if b < 0:
            raise RuntimeError

    stacked = torch.stack(inputs, dim=0)
    largest, _ = torch.topk(stacked, b, 0)
    neg_smallest, _ = torch.topk(-stacked, b, 0)
    new_stacked = torch.cat([stacked, -largest, neg_smallest]).sum(0)
    new_stacked /= len(inputs) - 2 * b
    return new_stacked

def agg_krum(inputs, n_byz):
    n_cl = len(inputs)
    tensor = inputs
    squared_dists = torch.cdist(tensor, tensor).square()
    topk_dists, _ = squared_dists.topk(k=n_cl - n_byz -1, dim=-1, largest=False, sorted=False)
    scores = topk_dists.sum(dim=-1)
    _, candidate_idxs = scores.topk(k=n_cl - n_byz, dim=-1, largest=False)

    agg_tensor = tensor[candidate_idxs].mean(dim=0)
    krum_passed = candidate_idxs > (n_cl - n_byz) - 1
    return agg_tensor


def agg_bulyan(inputs, n_byz):
    n_cl = len(inputs)
    tensor = inputs
    client_tensor = torch.tensor(list(range(n_cl))).to(tensor.device)
    squared_dists = torch.cdist(tensor, tensor).square()
    topk_dists, _ = squared_dists.topk(k=n_cl - n_byz - 1, dim=-1, largest=False, sorted=False)
    scores = topk_dists.sum(dim=-1)
    _, krum_candidate_idxs = scores.topk(k=n_cl - 2 * n_byz, dim=-1, largest=False)
    krum_tensor = tensor[krum_candidate_idxs]
    krum_passed = (krum_candidate_idxs > (n_cl- n_byz) -1).sum() / n_byz
    #print(krum_passed)
    med, _ = krum_tensor.median(dim=0)
    dist = (krum_tensor - med).abs()
    tr_updates, _ = dist.topk(k=n_cl - 4 * n_byz, dim=0, largest=False)
    agg_tensor = tr_updates.mean(dim=0)
    return agg_tensor


def base_aggr(inputs,aggr_type,m):
    if aggr_type =='bulyan':
        aggr_result = agg_bulyan(inputs,m)
    elif aggr_type == 'krum':
        aggr_result = agg_krum(inputs,m)
    return aggr_result


class Gas(_BaseAggregator):
    def __init__(self,n,m,p,aggr):
        self.n = n
        self.m = m
        self.p = p
        self.aggr = aggr
        super(Gas,self).__init__()

    def split(self, inputs):
        device = inputs[0].device
        d = inputs[0].numel()
        shuffled_dims = torch.randperm(d).to(device)
        p = self.p
        partition = torch.chunk(shuffled_dims, chunks=p)
        stacked = torch.stack(inputs, dim=0)
        groups = [stacked[:, partition_i] for partition_i in partition]
        return groups


    def __call__(self, inputs):
        groups = self.split(inputs)
        identification_scores = torch.zeros(self.n)
        for group in groups:
            group_agg = base_aggr(group,self.aggr,self.m)
            group_scores = (group - group_agg).square().sum(dim=-1).sqrt().cpu()
            identification_scores += group_scores
        _, cand_idxs = identification_scores.topk(k=self.n - self.m, largest=False)
        stacked = torch.stack(inputs, dim=0)
        aggr = stacked[cand_idxs].mean(dim=0)
        return aggr