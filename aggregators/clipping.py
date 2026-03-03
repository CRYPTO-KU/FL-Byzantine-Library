import torch
from .base import _BaseAggregator
import math


class Clipping(_BaseAggregator):
    def __init__(self, tau, b=5,n_iter=1):
        self.tau = tau
        self.n_iter = n_iter
        super(Clipping, self).__init__()
        self.momentum = None
        self.b = b
        self.cos_sim = torch.nn.CosineSimilarity(dim=0)
        

    def clip(self, v):
        v_norm = torch.norm(v)
        if v_norm == 0:
            return v  # If norm is zero, return as is
        scale = min(1, self.tau / v_norm)
        return v * scale

    def clipped_val(self,v):
        v_norm = torch.norm(v).item()
        if v_norm == 0:
            return 0  # If norm is zero, no clipping needed
        scale = min(1, self.tau / v_norm)
        if scale ==1:
            return 0
        else:
            return scale

    def __call__(self, inputs):
        if self.momentum is None:
            self.momentum = torch.zeros_like(inputs[0])

        self.avg_clip = []
        self.post_clip_angle = []

        for _ in range(self.n_iter):
            self.avg_clip.extend([self.clipped_val(v - self.momentum) for v in inputs[-self.b:]])
            self.post_clip_angle.extend([self.calc_angle(self.clip(v - self.momentum), self.momentum) for v in inputs[-self.b:]])

            self.momentum = (
                sum(self.clip(v - self.momentum) for v in inputs) / len(inputs)
                + self.momentum
            )

        return torch.clone(self.momentum).detach()
    
    def get_attack_stats(self) -> dict:
        attacker_stats = {'CC-clipped':sum(self.avg_clip) / len(self.avg_clip),
                          'CC-PostAngle':sum(self.post_clip_angle) / len(self.post_clip_angle)}
        return attacker_stats

    def cosine_to_degree(self,cos_sim):
# Clamp value to valid range to avoid numerical errors
        cos_sim = max(min(cos_sim, 1.0), -1.0)
        angle_rad = math.acos(cos_sim)
        angle_deg = math.degrees(angle_rad)
        return angle_deg

    def calc_angle(self,vec1,vec2):
        cos = self.cos_sim(vec1,vec2)
        return self.cosine_to_degree(cos) 



