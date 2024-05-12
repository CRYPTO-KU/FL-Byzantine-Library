from .base import _BaseByzantine
import torch

class minsum(_BaseByzantine): ## This uses too much memory and computationally heavy.
    def __init__(self,n,m,*args,**kwargs):
        super().__init__(*args, **kwargs)
        self.m = m

    def omniscient_callback(self,benign_gradients):
        # Loop over good workers and accumulate their gradients
        stacked_gradients = torch.stack(benign_gradients, 1)
        mu = torch.mean(stacked_gradients, 1).to(self.device)
        stack2 = torch.stack(benign_gradients, 0)
        m = self.our_attack_score(stack2, mu, dev_type=self.args.pert_vec)
        self.adv_momentum = m


    def local_step(self,batch):
        return None

    def train_(self, embd_momentum=None):
        return None

    def our_attack_score(self,all_updates, model_re, dev_type='unit_vec'):

        if dev_type == 'unit_vec':
            deviation = model_re / torch.norm(model_re)  # unit vector, dir opp to good dir
        elif dev_type == 'sign':
            deviation = torch.sign(model_re)
        elif dev_type == 'std':
            deviation = torch.std(all_updates, 0)

        lamda = torch.Tensor([10.0]).float().to(self.device)
        threshold_diff = 1e-5
        lamda_fail = lamda
        lamda_succ = 0

        distances = []
        for update in all_updates:
            distance = torch.norm((all_updates - update), dim=1) ** 2
            distances = distance[None, :] if not len(distances) else torch.cat((distances, distance[None, :]), 0)

        scores = torch.sum(distances, dim=1)
        min_score = torch.min(scores)
        del distances

        while torch.abs(lamda_succ - lamda) > threshold_diff:
            mal_update = (model_re - lamda * deviation)
            distance = torch.norm((all_updates - mal_update), dim=1) ** 2
            score = torch.sum(distance)

            if score <= min_score:
                # print('successful lamda is ', lamda)
                lamda_succ = lamda
                lamda = lamda + lamda_fail / 2
            else:
                lamda = lamda - lamda_fail / 2

            lamda_fail = lamda_fail / 2

        # print(lamda_succ)
        mal_update = (model_re - lamda_succ * deviation)

        return mal_update
