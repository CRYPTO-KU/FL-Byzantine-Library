import numpy as np
import torch

from .base import _BaseAggregator


def _compute_scores(distances, i, n, f):
    """Compute scores for node i.
    Arguments:
        distances -- A 2D numpy array of pairwise squared distances.
        i {int} -- index of worker, starting from 0.
        n {int} -- total number of workers
        f {int} -- Total number of Byzantine workers.
    Returns:
        float -- krum distance score of i.
    """
    s = np.concatenate([distances[:i, i], distances[i, i+1:]])
    _s = np.sort(s)[: n - f - 2]
    return _s.sum()


def krum(distances, n, f):
    """Krum algorithm
    Arguments:
        distances -- A 2D numpy array of pairwise squared distances.
        n {int} -- Total number of workers.
        f {int} -- Total number of Byzantine workers.
    Returns:
        int -- Index of the selected worker.
    """
    if n < 1:
        raise ValueError(
            "Number of workers should be positive integer. Got {}.".format(f)
        )

    if 2 * f + 2 > n:
        raise ValueError("Too many Byzantine workers: 2 * {} + 2 >= {}.".format(f, n))

    scores = [(i, _compute_scores(distances, i, n, f)) for i in range(n)]
    sorted_scores = sorted(scores, key=lambda x: x[1])
    return sorted_scores[0][0]

def multi_krum(distances, n, f, m):
    """Multi_Krum algorithm
    Arguments:
        distances -- A 2D numpy array of pairwise squared distances.
        n {int} -- Total number of workers.
        f {int} -- Total number of Byzantine workers.
        m {int} -- Number of workers for aggregation.
    Returns:
        list -- A list indices of worker indices for aggregation. length <= m
    """
    if n < 1:
        raise ValueError(
            "Number of workers should be positive integer. Got {}.".format(f)
        )

    if m < 1 or m > n:
        raise ValueError(
            "Number of workers for aggregation should be >=1 and <= {}. Got {}.".format(
                m, n
            )
        )

    if 2 * f + 2 > n:
        raise ValueError("Too many Byzantine workers: 2 * {} + 2 >= {}.".format(f, n))

    scores = [(i, _compute_scores(distances, i, n, f)) for i in range(n)]
    sorted_scores = sorted(scores, key=lambda x: x[1])
    return list(map(lambda x: x[0], sorted_scores))[:m]


def pairwise_euclidean_distances(vectors):
    """Compute the pairwise squared euclidean distance using torch.cdist.
    Arguments:
        vectors {list} -- A list of vectors.
    Returns:
        numpy array -- A 2D array of squared distances.
    """
    stacked = torch.stack(vectors)
    # cdist returns L2 distances; square them for Krum scoring
    dist_matrix = torch.cdist(stacked, stacked, p=2) ** 2
    return dist_matrix.detach().cpu().numpy()


class Krum(_BaseAggregator):
    r"""
    This script implements KRUM and Multi-KRUM algorithms.
    Blanchard, Peva, Rachid Guerraoui, and Julien Stainer.
    "Machine learning with adversaries: Byzantine tolerant gradient descent."
    Advances in Neural Information Processing Systems. 2017.
    """

    def __init__(self, n, f, m=None,mk=True):
        self.n = n
        self.f = f
        self.m = m
        self.mk = mk ## Multi-Krum
        self.success = 1
        self.impact_ratio = 1
        super(Krum, self).__init__()

    def __call__(self, inputs):
        distances = pairwise_euclidean_distances(inputs)
        if not self.mk:
            selected_index = krum(distances, self.n, self.f)
            top_m_indices = [selected_index]
            #print(selected_index)
        else:
            top_m_indices = multi_krum(distances, self.n, self.f, self.m)

        byzantine_clients = np.arange(0, self.n)[-self.f:]
        bypassed = 0
        for cl in byzantine_clients:
            if cl in top_m_indices:
                bypassed += 1
        if self.f > 0:
            self.success = bypassed / self.f
            self.impact_ratio = bypassed / len(top_m_indices)
        else:
            self.success = 0
            self.impact_ratio = 0
        values = sum(inputs[i] for i in top_m_indices) / len(top_m_indices)
        return values

    def get_attack_stats(self) ->dict:
        return {'Krum-Bypassed':self.success,
                'Krum-Impact': self.impact_ratio
                }
