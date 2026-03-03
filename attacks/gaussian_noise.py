from client import *

class gaussian_noise_traitor(client):
    def local_step(self,batch):
        device = self.device
        x, y = batch
        x, y = x.to(device), y.to(device)
        zero_grad(self.model)
        logits = self.model(x)
        loss = self.criterion(logits, y)
        loss.backward()
        grad_std = get_grad_flattened(self.model,self.device).std()
        sigma = self.args.g_sigma if self.args.g_sigma is not None else grad_std
        noise = torch.randn_like(get_grad_flattened(self.model,self.device)) * sigma
        unflat_grad(self.model,get_grad_flattened(self.model,
                            self.device) + noise)
        self.step_sgd()