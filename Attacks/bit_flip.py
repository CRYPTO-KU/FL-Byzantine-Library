from client import *

class bit_flip_traitor(client):
    def local_step(self,batch):
        device = self.device
        x, y = batch
        x, y = x.to(device), y.to(device)
        zero_grad(self.model)
        logits = self.model(x)
        loss = self.criterion(logits, y)
        loss.backward()
        unflat_grad(self.model,-get_grad_flattened(self.model,
                            self.device))
        self.step_sgd()