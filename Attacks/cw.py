from client import *

'''
Customized Carlini & Wagner Approach for semi-untargeted attacking.

Objective is to make the model predict the second most probable class for each input.

'''
class cw_traitor(client):
    def local_step(self,batch):
        device = self.device
        x, y = batch
        x, y = x.to(device), y.to(device)
        zero_grad(self.model)
        logits = self.model(x)
        ps_logits = self.model(x)
        ps_labels = ps_logits.topk(2,dim=1)[1]
        traitor_labels = torch.zeros(0,device=device)
        for ps_label , true_label in zip(ps_labels,y):
            if true_label != ps_label[1]:
                traitor_labels = torch.cat((traitor_labels,ps_label[1]))
            else:
                traitor_labels = torch.cat((traitor_labels,ps_label[0]))
        loss = self.criterion(logits, traitor_labels)
        loss.backward()
        step_sgd(self.model, self.momentum, self.lr, self.args)
