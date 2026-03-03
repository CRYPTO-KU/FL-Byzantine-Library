import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenetv2, mobilenetv3
import pruners.synflow_layers as layers

class MNIST_NET(nn.Module):
    def __init__(self, norm, num_classes=10):
        super(MNIST_NET, self).__init__()
        self.conv1 = layers.Conv2d(1, 20, 5, 1)
        self.conv2 = layers.Conv2d(20, 50, 5, 1)
        self.fc1 = layers.Linear(4 * 4 * 50, 500)
        self.fc2 = layers.Linear(500, num_classes)

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
    def __init__(self, norm, num_classes=10):
        super(SimpleCifarNet, self).__init__()
        self.conv1 = layers.Conv2d(3, 64, 3, padding=1)
        self.bn1 = norm(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = layers.Conv2d(64, 128, 3, padding=1)
        self.bn2 = norm(128)
        self.conv3 = layers.Conv2d(128, 256, 3, padding=1)
        self.bn3 = norm(256)
        self.conv4 = layers.Conv2d(256, 512, 3, padding=1)
        self.bn4 = norm(512)
        self.fc1 = layers.Linear(512 * 2 * 2, 128)
        self.fc2 = layers.Linear(128, 256)
        self.fc3 = layers.Linear(256, 512)
        self.fc4 = layers.Linear(512, 1024)
        self.fc5 = layers.Linear(1024, num_classes)

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
    def __init__(self, norm, num_classes=10):
        super(SimpleCifarNetMoon, self).__init__()
        self.conv1 = layers.Conv2d(3, 64, 3, padding=1)
        self.bn1 = norm(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = layers.Conv2d(64, 128, 3, padding=1)
        self.bn2 = norm(128)
        self.conv3 = layers.Conv2d(128, 256, 3, padding=1)
        self.bn3 = norm(256)
        self.conv4 = layers.Conv2d(256, 512, 3, padding=1)
        self.bn4 = norm(512)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.l1 = layers.Linear(512, 512)
        self.l2 = layers.Linear(512, 256)

        # last layer
        self.l3 = layers.Linear(256, num_classes)

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
        return x, y


class moon_net(nn.Module):
    def __init__(self, norm, num_classes=10, input_dim=(16 * 5 * 5)):
        super(moon_net, self).__init__()
        self.conv1 = layers.Conv2d(3, 6, 5)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = layers.Conv2d(6, 16, 5)

        # for now, we hard coded this network
        # i.e. we fix the number of hidden layers i.e. 2 layers
        self.fc1 = layers.Linear(input_dim, 120)
        self.fc2 = layers.Linear(120, 84)
        # self.fc3 = layers.Linear(hidden_dims[1], output_dim)
        self.l1 = layers.Linear(84, 84)
        self.l2 = layers.Linear(84, 256)

        # last layer
        self.l3 = layers.Linear(256, num_classes)

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

        # x = self.fc3(x)
        return x, y


class SimpleCNN(nn.Module):
    def __init__(self, norm, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = layers.Conv2d(3, 6, 5)
        self.bn1 = norm(6)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = layers.Conv2d(6, 16, 5)
        self.bn2 = norm(16)
        self.fc1 = layers.Linear(16 * 5 * 5, 120)
        self.fc2 = layers.Linear(120, 84)
        self.fc3 = layers.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1,norm=nn.BatchNorm2d, option='B'):
        super(BasicBlock, self).__init__()
        self.conv1 = layers.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = norm(planes)
        self.conv2 = layers.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = norm(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     layers.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     norm(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, norm=nn.BatchNorm2d, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = layers.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], norm=norm,stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], norm=norm,stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], norm=norm,stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], norm=norm,stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = layers.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, norm, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride,norm))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class ResNet_3Layer(nn.Module):
    def __init__(self, block, num_blocks, norm=nn.BatchNorm2d, num_classes=10):
        super(ResNet_3Layer, self).__init__()

        _outputs = [32, 64, 128]
        self.in_planes = _outputs[0]
        self.conv1 = layers.Conv2d(3, _outputs[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm(_outputs[0])
        self.layer1 = self._make_layer(block, _outputs[0], num_blocks[0], stride=1,norm=norm)
        self.layer2 = self._make_layer(block, _outputs[1], num_blocks[1], stride=2,norm=norm)
        self.layer3 = self._make_layer(block, _outputs[2], num_blocks[2], stride=2,norm=norm)
        self.linear = layers.Linear(_outputs[2], num_classes)


    def _make_layer(self, block, planes, num_blocks, stride,norm):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride,norm))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        out = F.log_softmax(out, dim=1)
        return out

import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self,input_size=784,num_class=10):
        super().__init__()
        self.hidden1 = layers.Linear(input_size,120)
        self.hidden2 = layers.Linear(120,84)
        self.classifier = layers.Linear(84,num_class)

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
      layers.Linear(28 * 28, 64),
      nn.ReLU(),
      layers.Linear(64, 32),
      nn.ReLU(),
      layers.Linear(32, num_class)
    )


  def forward(self, x):
    return self.layers(x)


class MLP_big(nn.Module):

    def __init__(self,num_class):
        super(MLP_big, self).__init__()
        self.fc1 = layers.Linear(28 * 28, 512)
        self.fc3 = layers.Linear(512, num_class)
        self.droput = nn.Dropout(0.2)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.droput(x)
        #x = self.droput(x)
        x = self.fc3(x)
        return x

def get_synflow_net(args):
    labels = {'cifar10': 10,'svhn': 10,'mnist':10,'fmnist':10,'emnist-d':10,
              'emnist-l': 26,
              'emnist-b': 47,
              'cifar100': 100,
              'tiny_imagenet': 200}
    num_cls = labels.get(args.dataset_name)
    norm = layers.BatchNorm2d
    # First Network Architecture, then its parameters in order
    neural_networks = {'simplecifar': [SimpleCifarNet, norm, num_cls],
                       'simplecnn': [SimpleCNN, norm, num_cls],
                       'simplecifarmoon': [SimpleCifarNetMoon, norm, num_cls],
                       'moon_net':[moon_net, norm, num_cls],
                       'resnet8': [ResNet_3Layer, BasicBlock, [1, 1, 1], norm, num_cls],
                       'resnet20': [ResNet_3Layer, BasicBlock, [2, 2, 2], norm, num_cls],
                       'resnet9': [ResNet, BasicBlock, [1, 1, 1, 1], norm, num_cls],
                       'resnet18': [ResNet, BasicBlock, [2, 2, 2, 2], norm, num_cls],
                       'mnistnet':[MNIST_NET,norm,num_cls],
                       'mlp_big':[MLP_big,num_cls],
                       'mlp_small': [MLP_small,num_cls],
                       }
    nn_name = args.nn_name.lower()
    try:
        network = neural_networks[nn_name]
    except:
        print('Available Neural Networks')
        print(neural_networks.keys())
        raise ValueError
    net = network[0](*network[1:])
    init_weights(net, args)
    return net

def kaiming_normal_init(m):  ## improves convergence at <200 comm rounds
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        # nn.init.xavier_normal(m.weight)
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        nn.init.constant_(m.bias, 0)


def init_weights(net, args):
    if args.weight_init == 'kn':
        for m in net.modules():
            kaiming_normal_init(m)