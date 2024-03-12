import torchvision
import torchvision.transforms as transforms
import torch


def get_MNIST_dataset(root,download,**kwargs):

    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    trainset = torchvision.datasets.MNIST(train=True, transform=transform_train,root=root,download=download)

    testset = torchvision.datasets.MNIST(train=False, transform=transform_test,root=root,download=download)

    return trainset, testset

def get_EMNIST_dataset(root,download,split,**kwargs):

    transform_train = transforms.Compose([
        transforms.ToTensor()
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor()
    ])

    trainset = torchvision.datasets.EMNIST(train=True, transform=transform_train,root=root,download=download,split=split)

    testset = torchvision.datasets.EMNIST(train=False, transform=transform_test,root=root,download=download,split=split)

    return trainset, testset

def get_EMNIST47_dataset(root,download,**kwargs):
    return get_EMNIST_dataset(root,download,split='balanced')

def get_EMNIST26_dataset(root,download,**kwargs):
    return get_EMNIST_dataset(root,download,split='letters')
def get_EMNIST10_dataset(root,download,**kwargs):
    return get_EMNIST_dataset(root,download,split='digits')

def get_FMNIST_dataset(root,download,**kwargs):

    transform_train = transforms.Compose([
        transforms.ToTensor()
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor()
    ])

    trainset = torchvision.datasets.FashionMNIST(train=True, transform=transform_train,root=root,download=download)

    testset = torchvision.datasets.FashionMNIST(train=False, transform=transform_test,root=root,download=download)


    return trainset, testset

# if __name__ == '__main__':
#     #balanced, letters, digits
#     get_EMNIST_dataset()