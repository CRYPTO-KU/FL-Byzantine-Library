from client import *

class label_flip_traitor(client):
    def local_step(self,batch):
        num_label = 100 if self.args.dataset_name =='cifar100' else 10
        x, y = batch
        y = y.to(self.device)
        #if self.args.nestrov_attack:
            #clean_step(self.model,x,y,self.criterion,self.lr,self.args)
        new_labels = torch.ones_like(y).mul_(num_label-1) - y
        x, y = x.to(self.device), new_labels.to(self.device)
        logits = self.model(x)
        zero_grad(self.model)
        loss = self.criterion(logits, new_labels)
        loss.backward()
        self.step_sgd()