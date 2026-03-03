from .base import _BaseByzantine

class IPMAttack(_BaseByzantine):
    def __init__(self, eps, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epsilon = eps

    def omniscient_callback(self,benign_gradients):
        # Loop over good workers and accumulate their gradients
        self.adv_momentum = -self.epsilon * (sum(benign_gradients)) / len(benign_gradients)

    def local_step(self,batch):
        return None

    def train_(self, embd_momentum=None):
        return None