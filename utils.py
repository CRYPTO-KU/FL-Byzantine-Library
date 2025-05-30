import random
import torch
import numpy as np
import datetime
import matplotlib.pyplot as plt
import os
from data_loader import DatasetSplit
import torch.nn.functional as F
import math
import torch.nn as nn
import nn_classes


def pull_model(model_user, model_server):
    with torch.no_grad():
        for param_user, param_server in zip(model_user.parameters(), model_server.parameters()):
            param_user.data = param_server.data[:] + 0
    return None


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def zero_grad(model):
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()


def disable_grad(model):
    for p in model.parameters():
        if p.grad is not None:
            p.grad.requires_grad_(False)


def initialize_zero(model):
    for param in model.parameters():
        param.data.zero_()
    return None


def get_grad_flattened(model, device):
    grad_flattened = torch.empty(0).to(device)
    for p in model.parameters():
        if p.requires_grad:
            a = p.grad.data.flatten().to(device)
            grad_flattened = torch.cat((grad_flattened, a), 0)
    return grad_flattened


def mean_vectors(vectors):
    values = torch.stack(vectors, dim=0).mean(dim=0)
    return values


def get_model_flattened(model, device):
    model_flattened = torch.empty(0).to(device)
    for p in model.parameters():
        a = p.data.flatten().to(device)
        model_flattened = torch.cat((model_flattened, a), 0)
    return model_flattened


def unflat_model(model, model_flattened):
    i = 0
    for p in model.parameters():
        temp = model_flattened[i:i + p.data.numel()]
        p.data = temp.reshape(p.data.size())
        i += p.data.numel()
    return None


def unflat_grad(model, grad_flattened):
    i = 0
    for p in model.parameters():
        if p.requires_grad:
            temp = grad_flattened[i:i + p.grad.data.numel()]
            p.grad.data = temp.reshape(p.data.size())
            i += p.data.numel()
    return None


def step_sgd(model, momentum, lr, args):
    last_ind = 0
    grad_mult = 1 - args.Lmomentum if args.one_sum_momentum else 1
    for p in model.parameters():
        if p.requires_grad:
            d_p = p.grad
            if args.wd != 0:
                d_p = d_p.add(p.data, alpha=args.wd)

            if momentum is None:
                buf = torch.clone(d_p).detach()
            else:
                length, dims = d_p.numel(), d_p.size()
                buf = momentum[last_ind:last_ind + length].view(dims).detach()
                buf.mul_(args.Lmomentum)
                buf.add_(torch.clone(d_p).detach(), alpha=grad_mult)
                if not args.embd_momentum:
                    momentum[last_ind:last_ind + length] = buf.flatten()  # update buffer
                last_ind += length
            if args.nesterov:
                d_p = d_p.add(buf, alpha=args.Lmomentum)
            else:
                d_p = buf
            p.data.add_(d_p, alpha=-lr)


def evaluate_accuracy(model, testloader, device):
    """Calculates the accuracy of the model"""
    correct = 0
    total = 0
    # model.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


def save_results(args, sparse_mask, **kwargs):
    sim_id = random.randint(1, 99999)
    attack = args.attack
    if attack == 'sparse':
        attack += '_Cfg{}'.format(args.sparse_cfg)
    aggr = args.aggr
    if aggr == 'cc':
        aggr = '{}_Tau_{}'.format(aggr, args.tau)
    momentum = args.Lmomentum
    dataset = '{}_{}'.format(args.dataset_name, args.dataset_dist)
    delta = int(args.traitor * args.num_client) if args.traitor < 1 else int(args.traitor)
    path_ = 'ATK_{}-Def_{}-dist_{}-B_{}-Z_{}-L_{}-D_{}-{}'.format(attack, aggr,
                                                                  dataset, momentum, args.z_max, args.lamb, delta,
                                                                  sim_id)
    path = 'Results'
    if args.aggr == 'avg' and args.traitor == 0:
        path_ = 'Baseline-{}-B_{}-{}'.format(dataset, momentum, sim_id)
    elif args.traitor == 0:
        path_ = '{}-No_Attacker-{}-B_{}-{}'.format(aggr, dataset, momentum, sim_id)
    if not os.path.exists(path):
        os.mkdir(path)
    path = os.path.join(path, path_)
    os.mkdir(path)
    log(path, args, sim_id)
    vec_path = os.path.join(path, 'vecs')
    std_path = os.path.join(path, 'std')
    os.mkdir(vec_path)
    os.mkdir(std_path)
    if sparse_mask is not None:
        file_name = 'PruneMask-Plot.png'
        mask_name = 'Best_mask'
        net = nn_classes.get_net(args)
        save = os.path.join(path, file_name)
        pruned_histogram(save, net, sparse_mask)
        mask_file = os.path.join(path, mask_name)
        mask_inds = torch.arange(0, sparse_mask.numel(), device=sparse_mask.device)
        mask_inds = mask_inds[sparse_mask > 0]
        torch.save(mask_inds.detach().cpu(), mask_file)
    for key, vals in kwargs.items():
        if vals is not None:
            x_ = list(range(1, args.global_epoch + 1))
            if isinstance(vals, int):
                mean_val, std = -1, -1
            else:
                mean_val, std = vals.mean(axis=0), vals.std(axis=0)
            std = np.around(std, decimals=3)
            plt.plot(x_, mean_val, label=key)
            plt.xlabel('Epoch')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            save = os.path.join(path, '{}_Plot.png'.format(key))
            plt.savefig(save, bbox_inches='tight')
            plt.clf()
            f = open(path + '/log.txt', 'a')
            f.write('Avg {} : {},    '
                    'STD  :   {}'.format(key, mean_val[-1], (std[-1])) + '\n')
            f.close()
            np_file = os.path.join(vec_path, key)
            np_file_std = os.path.join(std_path, key)
            np.save(np_file, mean_val)
            np.save(np_file_std, std)
    return None


def log(path, args, sim_id):
    n_path = path
    f = open(n_path + '/log.txt', 'w+')
    f.write('############## Args ###############' + '\n')
    l = 'sim_id : {} \n'.format(sim_id)
    f.write(l)
    for arg in vars(args):
        line = str(arg) + ' : ' + str(getattr(args, arg))
        f.write(line + '\n')
    f.write('############ Results ###############' + '\n')
    f.close()


def early_stop(accs):
    if len(accs) < 10:
        return False
    else:
        if (sum(accs[-8:]) / 8) < 80:
            return False
        else:
            return True


def bn_mask(net, device):
    import torch.nn as nn
    mask = torch.empty((0), device=device)
    for layer in net.modules():
        if isinstance(layer, nn.BatchNorm2d):
            mask_ = torch.ones_like(layer.weight.data.detach().flatten())
            mask = torch.cat((mask, mask_), dim=0)
            if layer.bias is not None:
                mask_bias = torch.ones_like(layer.bias.data.detach().flatten())
                mask = torch.cat((mask, mask_bias), dim=0)
        elif isinstance(layer, nn.Linear):
            mask_ = torch.zeros_like(layer.weight.data.detach().flatten())
            mask = torch.cat((mask, mask_), dim=0)
            if layer.bias is not None:
                mask_bias = torch.ones_like(layer.bias.data.detach().flatten())
                mask = torch.cat((mask, mask_bias), dim=0)
        elif isinstance(layer, nn.Conv2d):
            mask_ = torch.zeros_like(layer.weight.data.detach().flatten())
            mask = torch.cat((mask, mask_), dim=0)
            if layer.bias is not None:  ## if there is no bn
                mask_bias = torch.ones_like(layer.bias.data.detach().flatten())
                mask = torch.cat((mask, mask_bias), dim=0)
    return mask


def intout_layer_mask(net, device):
    import torch.nn as nn
    mask = torch.empty((0), device=device)
    for layer in net.modules():
        if isinstance(layer, nn.BatchNorm2d):
            mask_ = torch.zeros_like(layer.weight.data.detach().flatten())
            mask = torch.cat((mask, mask_), dim=0)
            if layer.bias is not None:
                mask_bias = torch.zeros_like(layer.bias.data.detach().flatten())
                mask = torch.cat((mask, mask_bias), dim=0)
        elif isinstance(layer, nn.Linear):
            mask_ = torch.zeros_like(layer.weight.data.detach().flatten())
            mask = torch.cat((mask, mask_), dim=0)
            if layer.bias is not None:
                mask_bias = torch.ones_like(layer.bias.data.detach().flatten())
                mask = torch.cat((mask, mask_bias), dim=0)
        elif isinstance(layer, nn.Conv2d):
            if first_conv:
                mask_ = torch.ones_like(layer.weight.data.detach().flatten())
                first_conv = False
            else:
                mask_ = torch.zeros_like(layer.weight.data.detach().flatten())
            mask = torch.cat((mask, mask_), dim=0)
            if layer.bias is not None:  ## if there is no bn
                mask_bias = torch.ones_like(layer.bias.data.detach().flatten())
                mask = torch.cat((mask, mask_bias), dim=0)
    return mask


def get_model_layers(net):
    import torch.nn as nn
    layers = []
    for layer in net.modules():
        if isinstance(layer, (nn.BatchNorm2d, nn.Linear, nn.Conv2d)):
            layers.append(layer.weight.detach().flatten().cpu().numpy())
            if layer.bias is not None:
                layers.append(layer.bias.detach().flatten().cpu().numpy())
    return layers


def get_layer_stats(layers):
    means = [np.mean(l) for l in layers]
    stds = [np.std(l) for l in layers]
    print(means)
    print(stds)
    return


class CustomBatchNorm2d(torch.nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1,
                 affine=True, track_running_stats=True):
        super(CustomBatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)

    def forward(self, input):
        self._check_input_dim(input)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates
        if self.training:
            mean = input.mean([0, 2, 3])
            # use biased var in train
            var = input.var([0, 2, 3], unbiased=False)
            n = input.numel() / input.size(1)
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean \
                                    + (1 - exponential_average_factor) * self.running_mean
                # update running_var with unbiased var
                self.running_var = exponential_average_factor * var * n / (n - 1) \
                                   + (1 - exponential_average_factor) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var

        input = (input - mean[None, :, None, None]) / (torch.sqrt(var[None, :, None, None] + self.eps))
        if self.affine:
            input = input * self.weight[None, :, None, None] + self.bias[None, :, None, None]
        return input


def check_dist(actuals, preds, device):
    cos = torch.nn.CosineSimilarity(dim=0)
    stacked_gradients1 = torch.stack(actuals, 1)
    mu = torch.mean(stacked_gradients1, 1).to(device)
    std = torch.std(stacked_gradients1, 1).to(device)
    stacked_gradients2 = torch.stack(preds, 1)
    mu2 = torch.mean(stacked_gradients2, 1).to(device)
    std2 = torch.std(stacked_gradients2, 1).to(device)
    mu_dif, std_dif = torch.norm(mu - mu2), torch.norm(std - std2)
    mu_angle, std_angle = cos(mu, mu2), cos(std, std2)
    print('L2 dist', 'mean:', mu_dif.item(), 'STD:', std_dif.item())
    print('cos similarity', 'mean:', mu_angle.item(), 'STD:', std_angle.item())


def estimate_lr(prev_global, ps_model, grads: list, device, args):
    prev = get_model_flattened(prev_global, device)
    ps = get_model_flattened(ps_model, device)
    dif = prev - ps
    updates = torch.stack(grads, dim=0)
    updates = torch.mean(updates, dim=0)
    pred = (dif / updates).mean().item()
    return pred


def get_pruning_dataset(args, dataset):
    if args.dataset_name == 'svhn':
        labels = dataset.labels
    else:
        try:  ## PyTorch 1.5.0+
            labels = dataset.targets
        except:  ## old Torch versions
            labels = dataset.train_labels

    labels = np.asarray(labels, dtype='int')
    num_sample = len(labels)
    split = args.prune_dataset_split
    prune_samples = int(num_sample * split)
    samples = np.random.choice(labels, prune_samples, replace=False)
    prune_dataset = DatasetSplit(dataset, samples)
    loader = torch.utils.data.DataLoader(prune_dataset, batch_size=args.prune_bs, shuffle=True)
    return loader


def angle_calculation(args, ref, momentum):
    def clip(v):
        v_norm = torch.norm(v)
        scale = min(1, args.tau / v_norm)
        return v * scale

    def get_angle(ref, pert):
        angle = math.degrees(math.acos(F.cosine_similarity(pert, ref, dim=0).item()))
        return angle

    angle1 = momentum - ref
    clipped = clip(angle1)
    norm = torch.norm(clipped)
    pre_clip = get_angle(ref, angle1)
    after_clip = get_angle(ref, clipped)
    return pre_clip, after_clip


def distance_calc(ref, momentums):
    def _compute_euclidean_distance(v1, v2):
        return (v1 - v2).norm()

    def get_angle(ref, momentum):
        angle = math.degrees(math.acos(F.cosine_similarity(momentum, ref, dim=0).item()))
        return angle


def pruned_histogram(loc, net, mask, exclude_bns=True):
    dims, flat_dims, names, colors = get_layer_dims(net)
    cum_dim = [0]
    cum_dim.extend(flat_dims)
    cum_dim = np.cumsum(cum_dim)
    mask = mask.detach().cpu().numpy()
    totals = []
    total_max = 0
    for i in range(len(cum_dim) - 1):
        b, e = cum_dim[i], cum_dim[i + 1]
        p = mask[b:e]
        size = len(p)
        pruned = np.sum(p)
        total = round((pruned / size) * 100, 1)
        totals.append(total)
        if 99 > total > total_max:
            total_max = total
    if exclude_bns:
        colors = np.asarray(colors)
        weights = colors != 'tab:red'
        colors = colors[weights]
        totals = np.asarray(totals)[weights]
        names = np.asarray(names)[weights]
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.bar(names, totals, color=colors)
    ax.set_ylabel('Left parameters %')
    ax.set_title('Pruning weights')
    ax.set_xlabel('Layers')
    ax.set_ylim([0, total_max + 5])
    plt.tight_layout()
    plt.savefig(loc)
    return


def get_layer_dims(net):
    dims = []
    names = []
    colors = []
    b, c, f = 1, 1, 1
    for layer in net.modules():
        if isinstance(layer, nn.BatchNorm2d):
            dims.append(layer.weight.shape)
            colors.append('tab:red')
            names.append('bnw{}'.format(b))
            if layer.bias is not None:
                dims.append(layer.bias.shape)
                names.append('bnb{}'.format(b))
                colors.append('tab:red')
            b += 1
        elif isinstance(layer, nn.Linear):
            dims.append(layer.weight.shape)
            colors.append('tab:blue')
            names.append('fcw{}'.format(f))
            if layer.bias is not None:
                dims.append(layer.weight.shape)
                names.append('fcb{}'.format(f))
                colors.append('tab:red')
            f += 1
        elif isinstance(layer, nn.Conv2d):
            dims.append(layer.weight.shape)
            colors.append('tab:blue')
            names.append('cw{}'.format(c))
            if layer.bias is not None:
                dims.append(layer.bias.shape)
                names.append('cb{}'.format(c))
                colors.append('tab:red')
            c += 1
    flat_dims = [np.prod(d) for d in dims]
    return dims, flat_dims, names, colors


def get_layer_dims2(net):
    dims = []
    is_crit = []
    for layer in net.modules():
        if isinstance(layer, (nn.BatchNorm2d, nn.Linear, nn.Conv2d)):
            dims.append(layer.weight.shape)
            if isinstance(layer, nn.BatchNorm2d):
                is_crit.append(True)
            else:
                is_crit.append(False)

            if layer.bias is not None:
                dims.append(layer.bias.shape)
                is_crit.append(True)
    flat_dims = [np.prod(d) for d in dims]
    flat_dims.insert(0, 0)
    locs = np.cumsum(flat_dims)
    return locs, is_crit


def get_layer_names(net):
    names = []
    bn_counter = 1
    fc_counter = 1
    conv_counter = 1
    for layer in net.modules():
        if isinstance(layer, nn.BatchNorm2d):
            names.append('bn' + str(bn_counter))
            if layer.bias is not None:
                names.append('bn' + str(bn_counter) + '_bias')
            bn_counter += 1
        elif isinstance(layer, nn.Linear):
            names.append('fc' + str(fc_counter))
            if layer.bias is not None:
                names.append('fc' + str(fc_counter) + '_bias')
            fc_counter += 1
        elif isinstance(layer, nn.Conv2d):
            names.append('conv' + str(conv_counter))
            if layer.bias is not None:
                names.append('conv' + str(conv_counter) + '_bias')
            conv_counter += 1
    return names
