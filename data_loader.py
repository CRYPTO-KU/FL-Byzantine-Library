from Datasets.RGB import *
from Datasets.BW import *
from torch.utils.data import Dataset
import numpy as np
import math
import random
from os import path,getcwd


kuac_mapper = {'cifar10':'cifar','cifar100':'cifar','tiny_imagenet':'tiny_imagenet_200_pytorch','fmnist':'Fashion-MNIST','mnist':'mnist'}

dataset_mapper = {'cifar10':get_cifar10_dataset,'cifar100':get_cifar100_dataset,
                  'emnist-b':get_EMNIST47_dataset,'emnist-l':get_EMNIST26_dataset,
                  'emnist-d':get_EMNIST10_dataset,
                  'svhn':get_svhn_dataset,'mnist':get_MNIST_dataset,
                  'fmnist':get_FMNIST_dataset,'tiny_imagenet':get_tiny_imagenet_dataset}

def get_dataset(args):
    dataset_name = args.dataset_name
    arg_dic = {'root': './data', 'download': True}
    trainset, testset = dataset_mapper[dataset_name.lower()](**arg_dic)
    return trainset,testset


def get_indices(trainset, args,test_set=None,num_cli_force=None):
    """returns the indices of sample for each worker in either iid, or non_iid manner provided in args"""
    if args.dataset_name == 'svhn':
        ''
        labels = trainset.labels
    else:
        try: ## PyTorch 1.5.0+
            labels = trainset.targets
        except: ## old Torch versions
            labels = trainset.train_labels

    labels = np.asarray(labels, dtype='int')
    nmbr_of_cls = max(labels) + 1
    image_per_clas = [np.sum(np.asarray(labels) == i) for i in range(nmbr_of_cls)]
    test_inds = None
    if args.dataset_dist == 'iid':
        inds = get_iid_index(labels, args,num_cli_force)
    elif args.dataset_dist == 'sort_part':
        inds = sort_and_part_dist(labels, args)
    elif args.dataset_dist == 'dirichlet':
        inds,dirichlet_vec = dirichlet_dist(labels, args,num_cli_force=num_cli_force)
        if test_set is not None:
            test_labels = test_set.targets
            test_labels = np.asarray(test_labels, dtype='int')
            test_inds,_ = dirichlet_dist(test_labels, args)
    elif args.dataset_dist == 'dirichlet_new':
        inds = dirichlet_new_dist(labels, args)
    else:
        raise ValueError('Dataset distribution can only be iid, sort_part, dirichlet, dirichlet_new')

    data_map = np.zeros((len(inds), nmbr_of_cls))
    for worker in inds:
        worker_inds = np.asarray(inds[worker], dtype='int')
        worker_labels = labels[worker_inds]
        for cls, cls_length in enumerate(image_per_clas):
            data_map[worker][cls] = np.sum(worker_labels == cls) / cls_length
    if test_set is not None:
        return inds,test_inds,data_map
    else:
        return inds, data_map


def sort_and_part_dist(labels, args):
    def partition(l, n):
        for i in range(0, len(l), n):
            yield l[i:i + n]

    class_per_usr = args.numb_cls_usr
    num_client = args.num_client
    num_samples = len(labels)
    num_class = max(labels) + 1
    inds_sorted = np.argsort(labels)
    class_size = int(num_samples / num_class)
    block_per_class = int(num_client * class_per_usr / num_class)
    data_per_block = int(class_size / block_per_class)
    excess = class_size - (data_per_block * block_per_class)
    all_datas = [[] for n in range(num_class)]
    seperated = list(partition(inds_sorted, class_size))
    sample_inds = {n: [] for n in range(num_client)}
    user_vec = np.repeat(class_per_usr, num_client)
    available_workers = np.arange(num_client)
    for i in range(num_class):
        class_partition = list(partition(seperated[i], data_per_block))
        if excess > 0:
            excess_block = class_partition[-1]
            class_partition.pop(-1)
            random.randint(0, block_per_class)
            for y, extra_data in enumerate(excess_block):
                class_partition[y % data_per_block] = np.append(class_partition[y % data_per_block], [extra_data])
        all_datas[i] = class_partition
    for label in range(num_class):
        remaining_label = num_class - label
        if remaining_label <= class_per_usr:
            selected_ = np.arange(num_client)[user_vec == remaining_label]
            available_workers_ = []
            for worker in available_workers:
                if worker not in selected_:
                    available_workers_.append(worker)
            choise = block_per_class - len(selected_)
            selected = np.random.choice(available_workers_, choise, replace=False)
            selected = np.append(selected, selected_)
        else:
            selected = np.random.choice(available_workers, block_per_class, replace=False)

        for client in selected:
            block_id = random.randint(0, len(all_datas[label]) - 1)
            block = all_datas[label][block_id]
            all_datas[label].pop(block_id)
            sample_inds[client] = np.concatenate([sample_inds[client], block.astype('int64')])
            sample_inds[client] = sample_inds[client].astype('int64')  # extra measures
            user_vec[int(client)] -= 1
        available_workers = np.arange(num_client)[user_vec > 0]
    return sample_inds


def dirichlet_dist(labels, args,dirichlet_vec=None,num_cli_force=None):
    num_cls = max(labels) + 1
    inds_sorted = np.argsort(labels)
    train_sorted = []
    num_client = args.num_client if num_cli_force is None else num_cli_force
    for i in range(num_cls):
        total_sample = np.sum(np.asarray(labels) == i)
        train_sorted.append(inds_sorted[i * total_sample:(i + 1) * total_sample])

    dirichlet = np.repeat(args.alpha, num_cls)
    if dirichlet_vec is None:
        dirichlet_vec = np.random.dirichlet(dirichlet, num_client)
    clas_dist = np.sum(dirichlet_vec, axis=0)

    indx_sample = {n: [] for n in range(num_client)}
    left_over_inds = []

    for cls in range(num_cls):  ##distribute data
        norm_val = 1 / clas_dist[cls]
        start_ind = 0
        cls_samples = len(train_sorted[cls])
        for i, worker in enumerate(dirichlet_vec):
            cls_alf = worker[cls]
            end_ind = start_ind + int(cls_samples * cls_alf * norm_val)
            indx_sample[i] = np.concatenate((indx_sample[i], train_sorted[cls][start_ind:end_ind]), axis=0)
            start_ind = end_ind
        if start_ind < len(train_sorted[cls]):
            left_over_inds = np.concatenate((left_over_inds, train_sorted[cls][start_ind:]))

    # distribute leftover inds
    np.random.shuffle(left_over_inds)
    dist = [len(indx_sample[sample]) for sample in indx_sample]
    ### line result in 0 div in low client numbers
    extra_sample = math.ceil(len(left_over_inds) / np.sum(np.asarray(dist) < (len(labels) / num_client)))
    ###
    given = 0
    for client in np.argsort(dist):  # prioritize client with fewer data
        if given + extra_sample < len(left_over_inds):
            indx_sample[client] = np.concatenate((indx_sample[client], left_over_inds[given:given + extra_sample]),
                                                 axis=0)
            given += extra_sample
        else:
            indx_sample[client] = np.concatenate((indx_sample[client], left_over_inds[given:]), axis=0)
            break

    for sample in indx_sample:
        indx_sample[sample] = np.asarray(indx_sample[sample], dtype='int')
    return indx_sample, dirichlet_vec

def dirichlet_new_dist(labels, args):
    alpha = args.alpha
    num_client = args.num_client

    num_classes = max(labels) + 1
    num_class_data = np.array([np.sum(np.array(labels) == i) for i in range(num_classes)])
    # assert sum(num_class_data) == len(labels)

    alpha_vec = np.repeat(alpha, num_classes)
    dirichlet_vec = np.random.dirichlet(alpha_vec, num_client)

    proportions = dirichlet_vec / dirichlet_vec.sum(axis=0, keepdims=True)
    # print(dirichlet_vec.shape, proportions.shape, proportions.sum(axis=0))

    indices = (proportions.cumsum(axis=0) * num_class_data[np.newaxis, :].repeat(num_client, axis=0))
    indices = indices.round().astype(int)

    inds_sorted = np.argsort(labels)

    res = {client_idx: [] for client_idx in range(num_client)}

    for cls_idx in range(num_classes):
        start_idx = int(num_class_data[:cls_idx].sum())
        last_idx = int(start_idx)
        for client_idx in range(num_client):
            until_idx = int(start_idx + indices[client_idx, cls_idx])
            indices_to_include = inds_sorted[last_idx:until_idx]
            res[client_idx].extend(indices_to_include)
            last_idx = until_idx

    return res


def get_iid_index(labels, args,num_cli_force=None):
    """Returns the indexes of samples for each user such that the distributions of data for each user
    have a iid distribution. Then equally splits
     the indexes for each user"""
    num_client = args.num_client if num_cli_force is None else num_cli_force
    num_samples = len(labels)
    num_sample_perworker = int(num_samples / num_client)
    inds = [*range(num_samples)]
    inds_split = np.random.choice(inds, [num_client, num_sample_perworker], replace=False)
    indx_sample = {n: [] for n in range(num_client)}
    for user in range(num_client):
        indx_sample[user] = list(inds_split[user])

    return indx_sample

def get_fl_trust_dataset(args,dataset):
    if args.dataset_name == 'svhn':
        labels = dataset.labels
    else:
        try:  ## PyTorch 1.5.0+
            labels = dataset.targets
        except:  ## old Torch versions
            labels = dataset.train_labels
    alpha = args.alpha

    num_classes = max(labels) + 1
    num_class_data = np.array([np.sum(np.array(labels) == i) for i in range(num_classes)])
    # assert sum(num_class_data) == len(labels)

    alpha_vec = np.repeat(alpha, num_classes)
    return


class DatasetSplit(Dataset):
    def __init__(self, dataset, indxs):
        self.dataset = dataset
        self.indxs = indxs

    def __len__(self):
        return len(self.indxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.indxs[item]]
        return image, label
