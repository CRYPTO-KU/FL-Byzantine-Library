import torch
from .base import _BaseAggregator


def _compute_euclidean_distance(v1, v2):
    return (v1 - v2).norm()


def smoothed_weiszfeld(weights, alphas, z, nu, T, b):
    m = len(weights)
    if len(alphas) != m:
        raise ValueError

    if nu < 0:
        raise ValueError

    malicious_betas = []
    benign_betas = []
    for t in range(T):
        betas = []
        bypassed = []
        for k in range(m):
            distance = _compute_euclidean_distance(z, weights[k])
            betas.append(alphas[k] / max(distance, nu))
        z = 0
        beta_m = betas[-b:]
        beta_m = [b.item() if isinstance(b, torch.Tensor) else b for b in beta_m]

        beta_b = betas[:-b]
        beta_b = [b.item() if isinstance(b, torch.Tensor) else b for b in beta_b]
        benign_betas.extend(beta_b)
        malicious_betas.extend(beta_m)
        for w, beta in zip(weights, betas):
            z += w * beta
        z /= sum(betas)
    return z, malicious_betas, benign_betas


class RFA(_BaseAggregator):
    r""""""

    def __init__(self, T, nu=1e-6):
        self.T = T
        self.nu = nu
        self.b = 5
        self.malicious_betas = []
        super(RFA, self).__init__()

    def __call__(self, inputs):
        alphas = [1 / len(inputs) for _ in inputs]
        z = torch.zeros_like(inputs[0])
        z, mal_betas, ben_betas = smoothed_weiszfeld(inputs, alphas, z=z, nu=self.nu, T=self.T, b=self.b)
        self.malicious_betas = mal_betas
        self.benign_betas = ben_betas
        return z

    def get_attack_stats(self):
        average_malicious = sum(self.malicious_betas) / len(self.malicious_betas) if self.malicious_betas else 0
        average_benign = sum(self.benign_betas) / len(self.benign_betas) if self.benign_betas else 0
        return {"average_malicious_beta": average_malicious, "average_benign_beta": average_benign}