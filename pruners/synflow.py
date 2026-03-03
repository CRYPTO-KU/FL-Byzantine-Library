import torch
import numpy as np
import pruners.synflow_layers as layers
from pruners.synflow_models import get_synflow_net
import torch.nn as nn

def masks(module):
    r"""Returns an iterator over modules masks, yielding the mask.
    """
    for name, buf in module.named_buffers():
        if "mask" in name:
            yield buf

def trainable(module):
    r"""Returns boolean whether a module is trainable.
    """
    return not isinstance(module, (layers.Identity1d, layers.Identity2d))

def prunable(module, batchnorm, residual):
    r"""Returns boolean whether a module is prunable.
    """
    isprunable = isinstance(module, (layers.Linear, layers.Conv2d))
    if batchnorm:
        isprunable |= isinstance(module, (layers.BatchNorm1d, layers.BatchNorm2d))
    if residual:
        isprunable |= isinstance(module, (layers.Identity1d, layers.Identity2d))
    #print(module, isprunable)
    return isprunable

def parameters(model):
    r"""Returns an iterator over models trainable parameters, yielding just the
    parameter tensor.
    """
    for module in filter(lambda p: trainable(p), model.modules()):
        for param in module.parameters(recurse=False):
            yield param

def masked_parameters(model, bias=False, batchnorm=False, residual=False):
    r"""Returns an iterator over models prunable parameters, yielding both the
    mask and parameter tensors.
    """
    for module in filter(lambda p: prunable(p, batchnorm, residual), model.modules()):
        for mask, param in zip(masks(module), module.parameters(recurse=False)):
            if param is not module.bias or bias is True:
                yield mask, param


class Pruner:
    def __init__(self, masked_parameters):
        self.masked_parameters = list(masked_parameters)
        self.scores = {}

    def score(self, model, loss, dataloader, device):
        raise NotImplementedError

    def _global_mask(self, sparsity):
        r"""Updates masks of model with scores by sparsity level globally.
        """
        # # Set score for masked parameters to -inf
        # for mask, param in self.masked_parameters:
        #     score = self.scores[id(param)]
        #     score[mask == 0.0] = -np.inf

        # Threshold scores
        global_scores = torch.cat([torch.flatten(v) for v in self.scores.values()])
        k = int((1.0 - sparsity) * global_scores.numel())
        if not k < 1:
            try:
                threshold, _ = torch.kthvalue(global_scores, k)
            except (RuntimeError, NotImplementedError):
                # Fallback for MPS or other backends that don't support kthvalue
                sorted_scores, _ = torch.sort(global_scores)
                threshold = sorted_scores[k-1]
            
            for mask, param in self.masked_parameters:
                score = self.scores[id(param)]
                zero = torch.tensor([0.]).to(mask.device)
                one = torch.tensor([1.]).to(mask.device)
                mask.copy_(torch.where(score <= threshold, zero, one))

    def _local_mask(self, sparsity):
        r"""Updates masks of model with scores by sparsity level parameter-wise.
        """
        for mask, param in self.masked_parameters:
            score = self.scores[id(param)]
            k = int((1.0 - sparsity) * score.numel())
            if not k < 1:
                try:
                    threshold, _ = torch.kthvalue(torch.flatten(score), k)
                except (RuntimeError, NotImplementedError):
                    # Fallback for MPS or other backends that don't support kthvalue
                    flattened_score = torch.flatten(score)
                    sorted_scores, _ = torch.sort(flattened_score)
                    threshold = sorted_scores[k-1]
                
                zero = torch.tensor([0.]).to(mask.device)
                one = torch.tensor([1.]).to(mask.device)
                mask.copy_(torch.where(score <= threshold, zero, one))

    def mask(self, sparsity, scope):
        r"""Updates masks of model with scores by sparsity according to scope.
        """
        if scope == 'global':
            self._global_mask(sparsity)
        if scope == 'local':
            self._local_mask(sparsity)

    @torch.no_grad()
    def apply_mask(self):
        r"""Applies mask to prunable parameters.
        """
        for mask, param in self.masked_parameters:
            param.mul_(mask)

    def alpha_mask(self, alpha):
        r"""Set all masks to alpha in model.
        """
        for mask, _ in self.masked_parameters:
            mask.fill_(alpha)

    # Based on https://github.com/facebookresearch/open_lth/blob/master/utils/tensor_utils.py#L43
    def shuffle(self):
        for mask, param in self.masked_parameters:
            shape = mask.shape
            perm = torch.randperm(mask.nelement())
            mask = mask.reshape(-1)[perm].reshape(shape)

    def invert(self):
        for v in self.scores.values():
            v.div_(v ** 2)

    def stats(self):
        r"""Returns remaining and total number of prunable parameters.
        """
        remaining_params, total_params = 0, 0
        for mask, _ in self.masked_parameters:
            remaining_params += mask.detach().cpu().numpy().sum()
            total_params += mask.numel()
        return remaining_params, total_params


class Rand(Pruner):
    def __init__(self, masked_parameters):
        super(Rand, self).__init__(masked_parameters)

    def score(self, model, loss, dataloader, device):
        for _, p in self.masked_parameters:
            self.scores[id(p)] = torch.randn_like(p)


class Mag(Pruner):
    def __init__(self, masked_parameters):
        super(Mag, self).__init__(masked_parameters)

    def score(self, model, loss, dataloader, device):
        for _, p in self.masked_parameters:
            self.scores[id(p)] = torch.clone(p.data).detach().abs_()


# Based on https://github.com/mi-lad/snip/blob/master/snip.py#L18
class SNIP(Pruner):
    def __init__(self, masked_parameters):
        super(SNIP, self).__init__(masked_parameters)

    def score(self, model, loss, dataloader, device):

        # allow masks to have gradient
        for m, _ in self.masked_parameters:
            m.requires_grad = True

        # compute gradient
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss(output, target).backward()

        # calculate score |g * theta|
        for m, p in self.masked_parameters:
            self.scores[id(p)] = torch.clone(m.grad).detach().abs_()
            p.grad.data.zero_()
            m.grad.data.zero_()
            m.requires_grad = False

        # normalize score
        all_scores = torch.cat([torch.flatten(v) for v in self.scores.values()])
        norm = torch.sum(all_scores)
        for _, p in self.masked_parameters:
            self.scores[id(p)].div_(norm)


# Based on https://github.com/alecwangcq/GraSP/blob/master/pruner/GraSP.py#L49
class GraSP(Pruner):
    def __init__(self, masked_parameters):
        super(GraSP, self).__init__(masked_parameters)
        self.temp = 200
        self.eps = 1e-10

    def score(self, model, loss, dataloader, device):

        # first gradient vector without computational graph
        stopped_grads = 0
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            output = model(data) / self.temp
            L = loss(output, target)

            grads = torch.autograd.grad(L, [p for (_, p) in self.masked_parameters], create_graph=False)
            flatten_grads = torch.cat([g.reshape(-1) for g in grads if g is not None])
            stopped_grads += flatten_grads

        # second gradient vector with computational graph
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            output = model(data) / self.temp
            L = loss(output, target)

            grads = torch.autograd.grad(L, [p for (_, p) in self.masked_parameters], create_graph=True)
            flatten_grads = torch.cat([g.reshape(-1) for g in grads if g is not None])

            gnorm = (stopped_grads * flatten_grads).sum()
            gnorm.backward()

        # calculate score Hg * theta (negate to remove top percent)
        for _, p in self.masked_parameters:
            self.scores[id(p)] = torch.clone(p.grad * p.data).detach()
            p.grad.data.zero_()

        # normalize score
        all_scores = torch.cat([torch.flatten(v) for v in self.scores.values()])
        norm = torch.abs(torch.sum(all_scores)) + self.eps
        for _, p in self.masked_parameters:
            self.scores[id(p)].div_(norm)


class SynFlow(Pruner):
    def __init__(self, masked_parameters):
        super(SynFlow, self).__init__(masked_parameters)

    def score(self, model, loss, dataloader, device):

        @torch.no_grad()
        def linearize(model):
            # model.double()
            signs = {}
            for name, param in model.state_dict().items():
                signs[name] = torch.sign(param)
                param.abs_()
            return signs

        @torch.no_grad()
        def nonlinearize(model, signs):
            # model.float()
            for name, param in model.state_dict().items():
                param.mul_(signs[name])

        signs = linearize(model)

        (data, _) = next(iter(dataloader))
        input_dim = list(data[0, :].shape)
        input = torch.ones([1] + input_dim).to(device)  # , dtype=torch.float64).to(device)
        output = model(input)
        torch.sum(output).backward()

        for _, p in self.masked_parameters:
            self.scores[id(p)] = torch.clone(p.grad * p).detach().abs_()
            p.grad.data.zero_()

        nonlinearize(model, signs)


def prune_loop(model, loss, pruner, dataloader, device, sparsity, schedule, scope, epochs,
               reinitialize=False, train_mode=False, shuffle=False, invert=False):
    r"""Applies score mask loop iteratively to a final sparsity level.
    """
    # Set model to train or eval mode
    model.train()
    if not train_mode:
        model.eval()

    # Prune model
    for epoch in (range(epochs)):
        pruner.score(model, loss, dataloader, device)
        if schedule == 'exp':
            sparse = sparsity ** ((epoch + 1) / epochs)
        elif schedule == 'linear':
            sparse = 1.0 - (1.0 - sparsity) * ((epoch + 1) / epochs)
        # Invert scores
        if invert:
            pruner.invert()
        pruner.mask(sparse, scope)

    # Reainitialize weights
    if reinitialize:
        model._initialize_weights()

    # Shuffle masks
    if shuffle:
        pruner.shuffle()

    # Confirm sparsity level
    remaining_params, total_params = pruner.stats()
    if np.abs(remaining_params - total_params * sparsity) >= 5:
        print("ERROR: {} prunable parameters remaining, expected {}".format(remaining_params, total_params * sparsity))

def get_synflow_mask(args,loader,device):
    model = get_synflow_net(args).to(device)
    bias = args.prune_bias
    batchnorm = args.prune_bn
    if args.prune_method == 'synflow':
        pruner = SynFlow(masked_parameters(model, bias,batchnorm,False))
    elif args.prune_method == 'grasp':
        pruner = GraSP(masked_parameters(model, bias, batchnorm, False))
    elif args.prune_method == 'snip':
        pruner = SNIP(masked_parameters(model, bias, batchnorm, False))
    loss = nn.CrossEntropyLoss()
    prune_loop(model, loss, pruner, loader, device, args.pruning_factor,
               args.mode, args.mask_scope, args.num_steps)
    masks = pruner.masked_parameters
    new_masks = []
    c = 0
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            p_mask, p = masks[c]
            new_masks.append(p_mask.flatten())
            c += 1
            if module.bias is not None:
                m = torch.ones_like(module.bias.flatten()).to(device)
                new_masks.append(m)
        elif isinstance(module, nn.BatchNorm2d):
            w = torch.ones_like(module.weight.flatten()).to(device)
            b = torch.ones_like(module.bias.flatten()).to(device)
            new_masks.append(w)
            new_masks.append(b)
    m = torch.empty((0)).to(device)
    for mask in new_masks:
        m = torch.cat((m, mask))
    return m
