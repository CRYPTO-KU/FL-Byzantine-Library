import torch.nn as nn
from utils import count_parameters
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self,input_size=784,num_class=10):
        super().__init__()
        self.hidden1 = nn.Linear(input_size,120)
        self.hidden2 = nn.Linear(120,84)
        self.classifier = nn.Linear(84,num_class)

    def forward(self,x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = self.classifier(x)
        return x

class MLP_small(nn.Module):
  '''
    Multilayer Perceptron.
  '''
  def __init__(self,num_class):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Flatten(),
      nn.Linear(28 * 28, 64),
      nn.ReLU(),
      nn.Linear(64, 32),
      nn.ReLU(),
      nn.Linear(32, num_class)
    )


  def forward(self, x):
    return self.layers(x)


class MLP_big(nn.Module):

    def __init__(self,num_class):
        super(MLP_big, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc3 = nn.Linear(512, num_class)
        self.droput = nn.Dropout(0.2)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.droput(x)
        #x = self.droput(x)
        x = self.fc3(x)
        return x

if __name__ == '__main__':
    net_1 = MLP_small(10)
    net_2 = MLP_big(10)
    print(count_parameters(net_1),count_parameters(net_2))
    for n in net_1.named_parameters():
        print(n)