import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn 
from utils import *
from copy import deepcopy
from typing import Optional, Dict, List, Tuple, Any, Callable
import argparse

class client():
    """
    A federated learning client that handles local training and model updates.
    
    This class represents a single participant in a federated learning system,
    responsible for local training on its dataset and communicating model updates
    with the central server.
    
    Attributes:
        id (int): Unique identifier for the client
        model (Optional[nn.Module]): The neural network model
        args (argparse.Namespace): Configuration arguments
        device (torch.device): Device for computation (CPU/GPU)
        data_size (int): Size of the local dataset
        bs (int): Batch size for training
        loader (DataLoader): DataLoader for the local dataset
        criterion (nn.CrossEntropyLoss): Loss function
        momentum (Optional[torch.Tensor]): Momentum buffer for optimization
        momentum2 (Optional[torch.Tensor]): Second momentum buffer (for Adam variants)
        local_steps (int): Number of local training steps
        lr (float): Learning rate
        mean_loss (float): Average loss from training
        omniscient (bool): Whether client has omniscient knowledge
        relocate (bool): Whether to relocate tensors
        step (int): Current optimization step
        opt_step (Callable): Optimization step function
        bn_stats (Optional[Dict]): Batch normalization statistics
        mask (Optional[torch.Tensor]): Sparsity mask for pruning
    """
    
    def __init__(self, id: int, dataset: Dataset, device: torch.device, args: argparse.Namespace, **kwargs: Any) -> None:
        """
        Initialize a federated learning client.
        
        Args:
            id: Unique identifier for the client
            dataset: Local dataset for training
            device: Computation device (CPU/GPU)
            args: Configuration arguments containing training parameters
            **kwargs: Additional keyword arguments
        """
        self.id = id
        self.model = None
        self.args = args
        self.device = device
        self.data_size = len(dataset)
        self.bs = args.bs if args.bs < self.data_size else self.data_size
        self.loader = DataLoader(dataset, batch_size=self.bs, shuffle=True,
                                  num_workers=getattr(args, 'num_workers', 0))
        self.criterion = nn.CrossEntropyLoss()
        self.momentum = None
        self.momentum2 = None
        self.local_steps = args.localIter
        self.lr = args.lr
        self.mean_loss = 0
        self.omniscient = False
        self.relocate = False
        self.step = 0
        self.opt_step = self.get_optim(args)
        self.bn_stats = None
        self.mask = None

    def local_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> None:
        """
        Perform a single local training step.
        
        Args:
            batch: A tuple containing input data and labels (x, y)
        """
        device = self.device
        x, y = batch
        x, y = x.to(device), y.to(device)
        zero_grad(self.model)
        logits = self.model(x)
        loss = self.criterion(logits, y)
        loss.backward()
        if self.args.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
        self.mean_loss = loss.item()
        self.opt_step()

    def train_(self, net: Optional[nn.Module] = None, embd_momentum: Optional[List[float]] = None) -> None:
        """
        Perform local training for the specified number of steps.
        
        Args:
            net: Optional pre-trained model to start training from
            embd_momentum: Optional embedded momentum values to initialize with
        """
        iterator = iter(self.loader)
        if self.momentum is not None:
            self.momentum = self.momentum.to(self.device)
        if net is not None:
            self.model = deepcopy(net)
            if self.bn_stats is not None:
                self.set_bn_stats(self.model)

        if embd_momentum is not None:
            self.momentum = torch.tensor(embd_momentum, device=self.device)
        elif self.momentum is None:
            model_size = count_parameters(self.model)
            self.momentum = torch.zeros(model_size, device=self.device)
        for i in range(self.local_steps):
            batch = iterator.__next__()
            self.local_step(batch)
        self.momentum = self.momentum.to('cpu')
        if net is not None:
            self.get_bn_stats(self.model)
            del self.model

    def get_bn_stats(self, net: nn.Module) -> None:
        """
        Extract and store batch normalization statistics from the model.
        
        Args:
            net: Neural network model to extract statistics from
        """
        if self.args.norm_type == 'bn':
            stats = {'mean':[],'var':[]}
            for mod, layer in net.named_modules():
                if isinstance(layer, nn.BatchNorm2d):
                    stats['mean'].append(deepcopy(layer.running_mean))
                    stats['var'].append(deepcopy(layer.running_var))
        else :
            stats = None
        self.bn_stats  = stats

    def set_bn_stats(self, net: nn.Module) -> None:
        """
        Apply stored batch normalization statistics to the model.
        
        Args:
            net: Neural network model to apply statistics to
        """
        c = 0
        for mod, layer in net.named_modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.running_mean = self.bn_stats['mean'][c]
                layer.running_var = self.bn_stats['var'][c]
                c += 1

    def update_mask(self, mask: torch.Tensor) -> None:
        """
        Update the sparsity mask for model pruning.
        
        Args:
            mask: Tensor representing the sparsity mask
        """
        self.mask = deepcopy(mask)
        
    def get_grad(self) -> torch.Tensor:
        """
        Get the gradient tensor based on the optimization algorithm.
        
        Returns:
            Processed gradient tensor (momentum for SGD, bias-corrected for Adam)
        """
        if self.args.opt == 'sgd':
            return torch.clone(self.momentum).detach()
        else: #for adams
            eps = 1e-08
            beta1, beta2 = self.args.betas
            new_moment = self.momentum.clone().detach() / (1- beta1**self.step)
            moment2 = self.momentum2.clone().detach() / (1- beta2 ** self.step)
            return new_moment / (torch.sqrt(moment2) + eps)

    def update_model(self, net_ps: nn.Module) -> None:
        """
        Update the client's model with new parameters.
        
        Args:
            net_ps: Neural network with new parameters to update with
        """
        if self.model is None:
            self.model = deepcopy(net_ps)
            self.model.train()
        else:
            pull_model(self.model, net_ps)

    def lr_step(self) -> None:
        """
        Perform a learning rate decay step by reducing the learning rate by a factor of 0.1.
        """
        self.lr *= .1

    def get_optim(self, args: argparse.Namespace) -> Callable[[], None]:
        """
        Get the appropriate optimization step function based on configuration.
        
        Args:
            args: Configuration arguments containing optimizer type
            
        Returns:
            Optimization step function
            
        Raises:
            NotImplementedError: If optimizer type is not supported
        """
        if args.opt == 'sgd':
            return self.step_sgd
        elif args.opt == 'adam':
            return self.step_adam
        elif args.opt == 'adamw': # if local iter is 1, regularization has no impact
            return self.step_adamw
        else:
            raise NotImplementedError('Invalid optimiser name')

    def step_sgd(self) -> None:
        """
        Perform one SGD optimization step with momentum support.
        
        Implements stochastic gradient descent with optional momentum,
        weight decay, and Nesterov acceleration.
        """
        args = self.args
        last_ind = 0
        grad_mult = 1 - args.Lmomentum if args.worker_momentum else 1
        for p in self.model.parameters():
            if p.requires_grad:
                d_p = p.grad
                if args.wd != 0:
                    d_p = d_p.add(p.data, alpha=args.wd)

                if self.momentum is None:
                    buf = torch.clone(d_p).detach()
                else:
                    length, dims = d_p.numel(), d_p.size()
                    buf = self.momentum[last_ind:last_ind + length].view(dims).detach()
                    buf.mul_(args.Lmomentum)
                    buf.add_(torch.clone(d_p).detach(), alpha=grad_mult)
                    if not args.embd_momentum:
                        self.momentum[last_ind:last_ind + length] = buf.flatten()  # update buffer
                    last_ind += length
                if args.nesterov:
                    d_p = d_p.add(buf, alpha=args.Lmomentum)
                else:
                    d_p = buf
                p.data.add_(d_p, alpha=-self.lr)

    def step_adam(self) -> None:
        """
        Perform one Adam optimization step.
        
        Implements the Adam optimization algorithm with bias correction
        for both first and second moment estimates.
        """
        last_ind = 0
        args = self.args
        eps = 1e-08
        self.step += 1
        if self.momentum2 is None:
            model_size = count_parameters(self.model)
            self.momentum = torch.zeros(model_size, device=self.device)
            self.momentum2 = torch.zeros(model_size, device=self.device)
        for p in self.model.parameters():
            if p.requires_grad:
                d_p = p.grad
                if args.wd != 0:
                    d_p = d_p.add(p.data, alpha=args.wd)
                length, dims = d_p.numel(), d_p.size()
                buf1 = self.momentum[last_ind:last_ind + length].view(dims).detach()
                buf2 = self.momentum2[last_ind:last_ind + length].view(dims).detach()
                m_t = buf1.mul(args.betas[0]) + d_p.mul(1-args.betas[0])
                v_t = buf2.mul(args.betas[1]) + torch.pow(d_p,2).mul(1-args.betas[1])
                self.momentum[last_ind:last_ind + length] = m_t.flatten()
                self.momentum2[last_ind:last_ind + length] = v_t.flatten()
                last_ind += length
                mt_h = m_t.div(1 - (args.betas[0]**self.step))
                vt_h = v_t.div(1 - (args.betas[1]**self.step))
                update = mt_h.div(torch.sqrt(vt_h)+eps)
                p.data.add_(update, alpha=-self.lr)


    def step_adamw(self) -> None:
        """
        Perform one AdamW optimization step.
        
        Implements the AdamW optimization algorithm which decouples
        weight decay from the gradient-based update.
        """
        args = self.args
        last_ind = 0
        eps = 1e-08
        self.step += 1
        if self.momentum is None:
            model_size = count_parameters(self.model)
            self.momentum = torch.zeros(model_size, device=self.device)
            self.momentum2 = torch.zeros(model_size, device=self.device)
        for p in self.model.parameters():
            if p.requires_grad:
                d_p = p.grad
                p.data.add_(p.data, alpha=args.wd * -self.lr)
                length, dims = d_p.numel(), d_p.size()
                buf1 = self.momentum[last_ind:last_ind + length].view(dims).detach()
                buf2 = self.momentum2[last_ind:last_ind + length].view(dims).detach()
                m_t = buf1.mul(args.betas(0)) + d_p.mul(1-args.betas(0))
                v_t = buf2.mul(args.betas(1)) + torch.pow(d_p,2).mul(1-args.betas(1))
                self.momentum[last_ind:last_ind + length] = m_t.flatten()
                self.momentum2[last_ind:last_ind + length] = v_t.flatten()
                last_ind += length
                mt_h = m_t.div(1 - torch.pow(args.betas(0), self.step))
                vt_h = v_t.div(1 - torch.pow(args.betas(1), self.step))
                update = mt_h.div(torch.sqrt(vt_h)+eps)
                p.data.add_(update, alpha=-self.lr)

