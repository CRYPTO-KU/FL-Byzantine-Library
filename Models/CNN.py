import torch.nn as nn
import torch.nn.functional as F
from torchvision. models import mobilenetv2,mobilenetv3


class MNIST_NET(nn.Module):
    def __init__(self,norm,num_classes=10):
        super(MNIST_NET, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class SimpleCifarNet(nn.Module):
    def __init__(self,norm,num_classes=10):
        super(SimpleCifarNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = norm(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = norm(128)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3 = norm(256)
        self.conv4 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn4 = norm(512)
        self.fc1 = nn.Linear(512 * 2 * 2, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 512)
        self.fc4 = nn.Linear(512, 1024)
        self.fc5 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = x.view(-1, 512 * 2 * 2)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x

class SimpleCifarNetMoon(nn.Module):
    def __init__(self,norm,num_classes=10):
        super(SimpleCifarNetMoon, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = norm(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = norm(128)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3 = norm(256)
        self.conv4 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn4 = norm(512)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.l1 = nn.Linear(512, 512)
        self.l2 = nn.Linear(512, 256)

        # last layer
        self.l3 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.avgpool(x)
        x = x.view(-1, 512)
        x = F.relu(self.l1(x))
        x = self.l2(x)
        y = self.l3(x)
        return x,y

class moon_net(nn.Module):
    def __init__(self, norm, num_classes=10, input_dim=(16 * 5 * 5)):
        super(moon_net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        # for now, we hard coded this network
        # i.e. we fix the number of hidden layers i.e. 2 layers
        self.fc1 = nn.Linear(input_dim, 120)
        self.fc2 = nn.Linear(120, 84)
        #self.fc3 = nn.Linear(hidden_dims[1], output_dim)
        self.l1 = nn.Linear(84, 84)
        self.l2 = nn.Linear(84, 256)

        # last layer
        self.l3 = nn.Linear(256, num_classes)

    def forward(self, x):

        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.l1(x)
        x = F.relu(x)
        x = self.l2(x)

        y = self.l3(x)

        #x = self.fc3(x)
        return x,y

class SimpleCNN(nn.Module):
    def __init__(self, norm,num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.bn1 = norm(6)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2 = norm(16)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class MobileNet(mobilenetv2.MobileNetV2):
    def __init__(self, norm, num_classes=10):
        super(MobileNet, self).__init__(num_classes=num_classes,norm_layer=norm)

    def forward(self, x):
        x = super(MobileNet, self).forward(x)
        return x
