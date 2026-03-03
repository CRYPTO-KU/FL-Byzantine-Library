from typing import List, Optional, Any, Tuple
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from aggregators.base import _BaseAggregator
from attacks.base import _BaseByzantine
from aggregators.Bucketing import Bucketing
from client import client  # assuming you have a Client class
from copy import deepcopy
from utils import count_parameters,evaluate_accuracy, update_model, get_grad_flattened
import argparse


class FL:
    """
    Federated Learning coordinator that manages training across multiple clients.
    
    This class orchestrates federated learning by coordinating training between
    benign and malicious clients, handling aggregation of gradients, and managing
    the global model updates.
    
    Attributes:
        args (argparse.Namespace): Configuration arguments
        total_samples (int): Total number of samples across all clients
        testloader (DataLoader): DataLoader for test dataset
        net_ps (nn.Module): Global neural network model
        device (torch.device): Device for computation (CPU/GPU)
        benign_clients (List[client]): List of honest clients
        malicious_clients (List[_BaseByzantine]): List of Byzantine/malicious clients
        malicious_ids (List[int]): Data indices of malicious clients
        bucket (Bucketing): Bucketing mechanism for gradient processing
        lr (float): Current learning rate
        aggr (_BaseAggregator): Aggregation strategy
        cli_rate (float): Client participation rate
        last_aggregate (Optional[torch.Tensor]): Last aggregated gradient
        epoch (float): Current training epoch
        comm_rounds (int): Number of communication rounds
        avg_train_loss (float): Average training loss
        psuedo_momentums (Optional[List[torch.Tensor]]): Pseudo momentum buffers
        divergence_rate (float): Threshold for detecting diverged clients
        num_diverged (float): Number of diverged clients
        std_tracker (Optional[torch.Tensor]): Standard deviation tracker for debugging
    """
    
    def __init__(self, 
                 args: argparse.Namespace,
                 net_ps: nn.Module,
                 total_samples: int,
                 test_set: Dataset,
                 benign_clients: List[client],
                 malicious_clients: List[_BaseByzantine],
                 aggr: _BaseAggregator,
                 bucket: Bucketing,
                 device: torch.device
                 ) -> None:
        """
        Initialize the Federated Learning coordinator.
        
        Args:
            args: Configuration arguments for the FL setup
            net_ps: Global neural network model
            total_samples: Total number of training samples across all clients
            test_set: Test dataset for evaluation
            benign_clients: List of honest/benign clients
            malicious_clients: List of Byzantine/malicious clients
            aggr: Aggregation strategy for combining gradients
            bucket: Bucketing mechanism for gradient processing
            device: Computation device (CPU/GPU/MPS)
        """
        self.args = args
        self.total_samples = total_samples
        self.testloader = DataLoader(test_set, batch_size=128, shuffle=False,
                                      num_workers=getattr(args, 'num_workers', 0))
        self.net_ps = net_ps
        self.device = device
        self.benign_clients = benign_clients
        self.malicious_clients = malicious_clients
        self.malicious_ids = [cl.id for cl in malicious_clients]
        self.bucket = bucket
        self.lr: float = args.lr
        self.aggr = aggr
        self.cli_rate: float = args.cl_part
        self.last_aggregate: Optional[torch.Tensor] = None
        self.epoch: float = 0
        self.comm_rounds: int = 0
        self.avg_train_loss: float = 0
        self.psuedo_momentums: Optional[List[torch.Tensor]] = None
        self.divergence_rate = 5
        self.num_diverged = 0
        self.std_tracker = None
        self.last_preds = None

    def cross_silo_step(self) -> List[torch.Tensor]:
        """
        Perform one cross-silo federated learning step with all clients participating.
        
        In cross-silo FL, all clients participate in each round. This method handles
        training of both benign and malicious clients, collects their gradients,
        and returns the combined gradient list.
        
        Returns:
            List of gradient tensors from all participating clients
        """
        for cl in self.benign_clients:
            cl.update_model(self.net_ps)
        for cl in self.malicious_clients:
            cl.update_model(self.net_ps)
        for cl in self.benign_clients:
            cl.train_()
        benign_grads = [cl.get_grad() for cl in self.benign_clients]
        # self.track_std(benign_grads)
        if self.args.MITM:
            byzantine_preds = benign_grads
        else:
            self.Byzantine_grad_preds()
            byzantine_preds = self.psuedo_momentums
        if len(self.malicious_clients) > 0:
            if self.malicious_clients[0].omniscient:
                self.malicious_clients[0].omniscient_callback(byzantine_preds)
                traitor_grads = [self.malicious_clients[0].adv_momentum for cl in
                                 self.malicious_clients]  # faster to do this than to call the function again
            else:
                for cl in self.malicious_clients:
                    cl.train_()
                traitor_grads = [cl.get_grad() for cl in self.malicious_clients]
            all_grads = benign_grads + traitor_grads
        else:
            all_grads = benign_grads
        self.avg_train_loss = sum([cl.mean_loss for cl in self.benign_clients]) / len(self.benign_clients)
        self.num_diverged = sum([cl.mean_loss > self.divergence_rate for cl in self.benign_clients]) / len(self.benign_clients)
        self.epoch += (len(all_grads) * self.args.localIter * self.benign_clients[0].bs) / self.total_samples
        #print(f"Epoch {self.epoch:.2f} | Avg Loss: {self.avg_train_loss:.4f}")
        return all_grads

    def cross_device_step(self) -> List[torch.Tensor]:
        """
        Perform one cross-device federated learning step with partial client participation.
        
        In cross-device FL, only a subset of clients participate in each round.
        This method randomly selects clients, handles their training, and returns
        the aggregated gradients.
        
        Returns:
            List of gradient tensors from selected participating clients
        """
        all_clients = self.benign_clients + self.malicious_clients
        all_clients = np.random.choice(all_clients, int(self.cli_rate * len(all_clients)), replace=False)
        #Client selection
        selected_benigns = [client for client in all_clients if client.id not in self.malicious_ids]
        selected_malicious = [client for client in all_clients if client.id in self.malicious_ids]
        #print(len(selected_benigns),len(selected_malicious))
        #training
        for cl in selected_benigns:
            cl.train_(net=self.net_ps)
        benign_grads = [cl.get_grad() for cl in selected_benigns]
        if len(selected_malicious) > 0:
            if selected_malicious[0].omniscient:
                selected_malicious[0].omniscient_callback(benign_grads)
                # If each attacker sends different updates this will need to change !
                traitor_grads = [selected_malicious[0].adv_momentum for cl in
                              selected_malicious]  # faster to do this than to call the function again
            else:
                for cl in selected_malicious:
                    cl.train_(net=self.net_ps)
                traitor_grads = [cl.get_grad() for cl in selected_malicious]
            all_grads = benign_grads + traitor_grads
        else:
            all_grads = benign_grads
        self.avg_train_loss = sum([cl.mean_loss for cl in selected_benigns]) / len(selected_benigns)
        self.epoch += (len(all_grads)* self.args.localIter * selected_benigns[0].bs) / self.total_samples
        return all_grads

    def __get_current_epoch__(self) -> float:
        """
        Get the current training epoch.
        
        Returns:
            Current epoch as a float value
        """
        return self.epoch

    def __update_lr__(self) -> None:
        """
        Update the learning rate for all clients.
        
        Applies learning rate decay to all benign and malicious clients
        and updates the coordinator's learning rate.
        """
        for cl in self.benign_clients:
            cl.lr_step()
        for cl in self.malicious_clients:
            cl.lr_step()
        self.lr = self.benign_clients[0].lr
        return

    def train(self) -> List[torch.Tensor]:
        """
        Execute one training round of federated learning.
        
        Determines whether to use cross-silo or cross-device approach based on
        client participation rate and executes the appropriate training step.
        
        Returns:
            List of gradient tensors from participating clients
        """
        self.comm_rounds +=1
        if self.args.cl_part < 1:
            all_grads = self.cross_device_step()
        else:
            all_grads = self.cross_silo_step()
        #self.std_analysis(all_grads)
        return all_grads

    def Byzantine_grad_preds(self) -> None:
        """
        Generate pseudo gradients for Byzantine clients to predict benign behavior.
        
        This method simulates benign client gradients by using malicious client
        data to create pseudo momentum buffers that can be used for attacks
        that require knowledge of benign gradients.
        """
        pred_per_byzantine = int((1-self.args.traitor) / self.args.traitor)
        if self.psuedo_momentums is None:
            self.psuedo_momentums = [torch.zeros(count_parameters(self.net_ps), device=self.device) for _ in self.benign_clients]
        net = deepcopy(self.net_ps).to(self.device)
        criterion = nn.CrossEntropyLoss()
        c = 0
        for cl in self.malicious_clients:
            for i,data in enumerate(cl.loader):
                net.zero_grad()
                if i == pred_per_byzantine:
                    break
                x, y = data
                x, y = x.to(self.device), y.to(self.device)
                logits = net(x)
                loss = criterion(logits, y)
                loss.backward()
                grad = get_grad_flattened(net,device=self.device)
                self.psuedo_momentums[c] = self.psuedo_momentums[c] * self.args.Lmomentum + grad * (1 - self.args.Lmomentum)
                c +=1


    def aggregate(self, all_grads: List[torch.Tensor]) -> None:
        """
        Aggregate gradients from all participating clients.
        
        Applies bucketing (if enabled) and then aggregates the gradients using
        the specified aggregation strategy. Updates the global aggregate.
        
        Args:
            all_grads: List of gradient tensors from participating clients
        """
        # Ensure all gradients are on the training device.
        # Client momentum is offloaded to CPU between rounds to save GPU memory,
        # so get_grad() returns CPU tensors. Move them back here centrally.
        all_grads = [g.to(self.device) for g in all_grads]
        
        if self.args.bucketing:
            all_grads = self.bucket.__call__(all_grads,self.last_aggregate)
        
        self.last_aggregate = self.aggr.__call__(all_grads)
        
        if self.malicious_clients[0].relocate:
                [cl.get_global_m(self.last_aggregate.clone()) for cl in self.malicious_clients]
        return
    
    def evaluate_accuracy(self) -> float:
        """
        Evaluate the accuracy of the global model on the test dataset.
        
        Returns:
            Test accuracy as a float value between 0 and 1
        """
        return evaluate_accuracy(self.net_ps, self.testloader, self.device)

    def update_global_model(self) -> None:
        """
        Update the global model parameters using the aggregated gradients.
        
        Applies the last aggregated gradient to the global model using
        the current learning rate.
        """
        update_model(self.net_ps, self.last_aggregate, self.lr, self.device)

    def final_prune_mask(self) -> Optional[torch.Tensor]:
        """
        Get the final pruning mask from the first malicious client.
        
        Returns:
            Pruning mask tensor if available, None otherwise
        """
        return self.malicious_clients[0].mask

    def get_aggr_success_info(self) -> Any:
        """
        Get attack success statistics from the aggregator.
        
        Returns:
            Attack statistics from the aggregation strategy
        """
        return self.aggr.get_attack_stats()
    

    def track_std(self, benign_grads: List[torch.Tensor]) -> None:
        """
        Track the standard deviation of benign gradients for debugging purposes.
        
        This method analyzes the variance in benign client gradients and tracks
        which gradient positions have the highest standard deviation across clients.
        
        Args:
            benign_grads: List of gradient tensors from benign clients
        """
        stacked_vals = torch.stack(benign_grads, dim=0)
        std = torch.std(stacked_vals, dim=0)
        target = self.args.pruning_factor
        _,locs = torch.topk(std, int(target * std.numel()), largest=True)
        if self.std_tracker is None:
            self.std_tracker = torch.zeros_like(std)
        self.std_tracker[locs] += 1