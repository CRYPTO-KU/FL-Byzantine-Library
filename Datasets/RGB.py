import torchvision
import torchvision.transforms as transforms
from os import path
def get_cifar10_dataset(root,download,**kwargs):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(train=True, transform=transform_train, root=root,download=download)

    testset = torchvision.datasets.CIFAR10(train=False, transform=transform_test,root=root,download=download)

    return trainset, testset


def get_cifar100_dataset(root,download,**kwargs):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
    ])

    # Normalize test set same as training set without augmentation
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
    ])

    trainset = torchvision.datasets.CIFAR100(train=True, transform=transform_train, root=root,download=download)

    testset = torchvision.datasets.CIFAR100(train=False, transform=transform_test, root=root,download=download)
    return trainset, testset


def get_svhn_dataset(root,download,**kwargs):
    mean = [0.4380, 0.4440, 0.4730]
    std = [0.1751, 0.1771, 0.1744]

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    trainset = torchvision.datasets.SVHN(split='train', transform=transform_train, root=root,download=download)

    testset = torchvision.datasets.SVHN(split='test', transform=transform_test, root=root,download=download)

    return trainset, testset


def get_tiny_imagenet_dataset(root,**kwargs):
    # https://drive.google.com/file/d/1U1DH9_eeJVkvoX0L1setzkn4rvRK_9Hf/view?usp=sharing
    # must be downloaded first than unzipped at the root location.
    if root == './data':
        train = path.join(root,'tiny-imagenet-200','train')
        test = path.join(root, 'tiny-imagenet-200', 'val')
    else:
        train = path.join(root, 'train')
        test = path.join(root,'val')

    tiny_mean = [0.48024578664982126, 0.44807218089384643, 0.3975477478649648]
    tiny_std = [0.2769864069088257, 0.26906448510256, 0.282081906210584]
    transform_train = transforms.Compose([
        transforms.RandomCrop(64, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(tiny_mean, tiny_std)])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(tiny_mean, tiny_std)])
    trainset = torchvision.datasets.ImageFolder(train,
                                                transform=transform_train)
    testset = torchvision.datasets.ImageFolder(test,
                                               transform=transform_test)

    return trainset, testset


def get_pcam_dataset(root,download,**kwargs):
    mean = [0.4380, 0.4440, 0.4730]
    std = [0.1751, 0.1771, 0.1744]

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    trainset = torchvision.datasets.PCAM(split='train', transform=transform_train, root=root,download=download)

    testset = torchvision.datasets.PCAM(split='test', transform=transform_test, root=root,download=download)

    return trainset, testset