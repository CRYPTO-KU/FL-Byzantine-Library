from Models.CNN import *
from Models.ResNet import *
from Models.RNN import RecurrentModel
from Models.VGG import VGG
from Models.MLP import *
from Models.Efficientnet import EfficientNetB0
from utils import count_parameters

num_groups_ = 32

def get_net(args):
    global num_groups_
    num_groups_ = args.num_groups
    norms = {'bn': nn.BatchNorm2d,
             'gn': GroupNorm,
             '-': NoneNorm, None: NoneNorm}
    labels = {'cifar10': 10,'svhn': 10,'mnist':10,'fmnist':10,'emnist-d':10,
              'emnist-l': 26,
              'emnist-b': 47,
              'cifar100': 100,
              'tiny_imagenet': 200}
    norm, num_cls = norms.get(args.norm_type), labels.get(args.dataset_name)
    # First Network Architecture, then its parameters in order
    neural_networks = {'simplecifar': [SimpleCifarNet, norm, num_cls],
                       'simplecnn': [SimpleCNN, norm, num_cls],
                       'simplecifarmoon': [SimpleCifarNetMoon, norm, num_cls],
                       'moon_net':[moon_net, norm, num_cls],
                       'resnet8': [ResNet_3Layer, BasicBlock, [1, 1, 1], norm, num_cls],
                       'resnet20': [ResNet_3Layer, BasicBlock, [2, 2, 2], norm, num_cls],
                       'resnet9': [ResNet, BasicBlock, [1, 1, 1, 1], norm, num_cls],
                       'resnet18': [ResNet, BasicBlock, [2, 2, 2, 2], norm, num_cls],
                       'vgg11': [VGG, '11', norm, num_cls],
                       'vgg13': [VGG, '13', norm, num_cls],
                       'vgg16': [VGG, '16', norm, num_cls],
                       'vgg19': [VGG, '19', norm, num_cls],
                       'mobilenet': [MobileNet,norm, num_cls],
                       'efficientnet':[EfficientNetB0,norm,num_cls],
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

class NoneNorm(nn.Module):
    def __init__(self, num_features):
        super(NoneNorm, self).__init__()

    def forward(self, x):
        return x

class GroupNorm(nn.GroupNorm):
    def __init__(self, num_channels):
        super().__init__(num_groups=num_groups_, num_channels=num_channels)

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

# if __name__ == '__main__':