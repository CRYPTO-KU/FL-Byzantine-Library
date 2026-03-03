import torch
import numpy as np
from .base import _BaseAggregator
from .krum import Krum, multi_krum, pairwise_euclidean_distances
from .trimmed_mean import TM
from .clipping import Clipping
from .rfa import RFA, smoothed_weiszfeld 


class HybridAggregator(_BaseAggregator):
    """
    Hybrid aggregator that combines multiple Byzantine-robust aggregation methods.
    
    This aggregator performs a multi-stage filtering and aggregation process:
    1. Each specified aggregator performs elimination/sanitization
    2. The intersection of selected clients from all aggregators is computed
    3. Final aggregation is performed using the last specified aggregator
    
    Args:
        n (int): Total number of clients
        m (int): Number of malicious/Byzantine clients
        aggregator_list (list of str): List of aggregator names to use sequentially
    """

    def __init__(self, n, m, f, tau, aggregator_list):
        super(HybridAggregator, self).__init__()
        self.n = n
        self.m = m  # Number of malicious clients
        self.f = f  # Byzantine tolerance parameter
        self.tau = tau
        self.aggregator_list = aggregator_list
        
        # Initialize individual aggregators
        self.aggregators = {}
        self._initialize_aggregators()
        
        # Statistics tracking
        self.elimination_stats = {}
        self.final_selection_stats = {}
        self.bypass_stats = {}
        
        # For Centered Clipping: maintain last aggregated result as reference
        self.last_aggregated_result = None
        self.iteration_count = 0
        
    def _initialize_aggregators(self):
        """Initialize all specified aggregators with appropriate parameters."""
        
        # Start with original n, m values
        current_n = self.n
        current_m = self.m  # Keep malicious count constant
        
        # print(f"[Hybrid] Initializing aggregators with cascading n values:")
        # print(f"[Hybrid] Original: n={self.n}, m={self.m}")
        
        for i, aggr_name in enumerate(self.aggregator_list):
            # print(f"[Hybrid] Step {i+1}: {aggr_name} with n={current_n}, m={current_m}")
            
            if aggr_name.lower() == 'multi_krum' or aggr_name.lower() == 'krum':
                # Multi-Krum parameters: automatically compute m = n - f - 2 for Byzantine tolerance
                krum_m = current_n - current_m - 2  # Byzantine-tolerant selection
                krum_m = max(1, min(krum_m, current_n))  # Ensure valid range
                
                self.aggregators[aggr_name] = Krum(
                    n=current_n, 
                    f=current_m, 
                    m=krum_m, 
                    mk=True  # Always use Multi-Krum for selection
                )
                
                # Update n for next aggregator (Multi-Krum selects krum_m clients)
                current_n = krum_m
                
            elif aggr_name.lower() == 'trimmed_mean' or aggr_name.lower() == 'tm':
                # Trimmed Mean parameters: automatically compute b based on Byzantine tolerance
                tm_b = min(current_m, (current_n - 1) // 2)  # Use current m, but ensure valid
                tm_b = max(0, min(tm_b, (current_n - 1) // 2))  # Ensure we don't trim everything
                
                self.aggregators[aggr_name] = TM(b=tm_b)
                
                # Update n for next aggregator (TM keeps n - 2*b clients)
                remaining_after_trim = current_n - 2 * tm_b
                current_n = max(1, remaining_after_trim)  # Ensure at least 1 client
                
            elif aggr_name.lower() == 'clipping' or aggr_name.lower() == 'cc':
                # Centered Clipping parameters: use default values for automatic operation
                tau = 1.0  # Default clipping threshold
                clipping_b = 5  # Default number of recent gradients for angle computation
                n_iter = 1  # Default number of iterations
                
                self.aggregators[aggr_name] = Clipping(tau=tau, b=clipping_b, n_iter=n_iter)
                
                # Clipping clips ALL inputs and uses them all - no client reduction
                # current_n stays the same
                
            elif aggr_name.lower() == 'rfa' or aggr_name.lower() == 'robust_federated_averaging':
                # RFA parameters: use default values for automatic operation
                T = 5  # Default number of iterations for smoothed Weiszfeld
                nu = 1e-6  # Default regularization parameter
                
                self.aggregators[aggr_name] = RFA(T=T, nu=nu)
                
                # RFA processes ALL inputs and uses them all - no client reduction
                # current_n stays the same
                
            else:
                raise ValueError(f"Unknown aggregator: {aggr_name}")
            
            # print(f"[Hybrid]   → Reduces n to {current_n} for next step")
        
        # print(f"[Hybrid] Final expected client count: {current_n}")
        
        # Store the progression for reference
        self.n_progression = self._compute_n_progression()
        # print(f"[Hybrid] Client count progression: {self.n_progression}")
    
    def _compute_n_progression(self):
        """Compute the expected progression of client counts through the pipeline."""
        progression = [self.n]
        current_n = self.n
        current_m = self.m
        
        for aggr_name in self.aggregator_list:
            if aggr_name.lower() in ['multi_krum', 'krum']:
                # Multi-Krum: automatically compute m = n - f - 2 (Byzantine-tolerant selection)
                krum_m = current_n - current_m - 2
                krum_m = max(1, min(krum_m, current_n))
                current_n = krum_m
                
            elif aggr_name.lower() in ['trimmed_mean', 'tm']:
                # Trimmed Mean: automatically compute b based on Byzantine tolerance
                tm_b = min(current_m, (current_n - 1) // 2)
                tm_b = max(0, min(tm_b, (current_n - 1) // 2))
                remaining_after_trim = current_n - 2 * tm_b
                current_n = max(1, remaining_after_trim)
                
            elif aggr_name.lower() in ['clipping', 'centered_clipping']:
                # Clipping clips ALL inputs and uses them all - no client reduction
                # current_n stays the same
                pass
                
            elif aggr_name.lower() in ['rfa', 'robust_federated_averaging']:
                # RFA processes ALL inputs and uses them all - no client reduction
                # current_n stays the same
                pass
            
            progression.append(current_n)
        
        return progression
    
    def clip(self, v):
        v_norm = torch.norm(v)
        if v_norm == 0:
            return v  # If norm is zero, return as is
        scale = min(1, self.tau / v_norm)
        return v * scale
    
    def clip_and_update(self, inputs: list, ref:torch.Tensor):
        grads = [self.clip(g - ref) + ref for g in inputs]
        return grads
    
    def _get_client_selection_multikrum(self, inputs, step_index=0):
        """Get client selection from Multi-Krum."""
        distances = pairwise_euclidean_distances(inputs)
        
        # Get the aggregator instance to use its configured parameters
        aggr_name = self.aggregator_list[step_index]
        krum_aggregator = self.aggregators[aggr_name]
        
        # Use the aggregator's configured parameters
        current_n = len(inputs)
        current_m = krum_aggregator.m  # Number of clients to select
        current_f = krum_aggregator.f  # Byzantine tolerance
        
        selected_indices = multi_krum(distances, current_n, current_f, current_m)
        return set(selected_indices)
    
    def _get_client_selection_trimmed_mean(self, inputs, step_index=0):
        """
        Get client selection from Trimmed Mean using coordinate-wise elimination.
        Performs proper trimmed mean by sorting each coordinate independently.
        """
        # Get the aggregator instance to use its configured parameters
        aggr_name = self.aggregator_list[step_index]
        tm_aggregator = self.aggregators[aggr_name]
        tm_b = tm_aggregator.b
        
        if tm_b is None or tm_b <= 0 or len(inputs) <= 2 * tm_b:
            # If no trimming needed, return all clients
            return set(range(len(inputs)))
        
        # Stack inputs for coordinate-wise processing
        stacked = torch.stack(inputs, dim=0)  # Shape: [num_clients, tensor_dims...]
        
        # For client selection approximation, we'll use a voting mechanism:
        # For each coordinate, identify which clients would be trimmed
        # Clients that are trimmed in many coordinates are more likely to be outliers
        
        # Flatten the tensor dimensions (except client dimension) for easier processing
        original_shape = stacked.shape
        if len(original_shape) > 2:
            # Reshape to [num_clients, -1] for processing
            stacked_flat = stacked.view(original_shape[0], -1)
        else:
            stacked_flat = stacked
        
        # Count how many coordinates each client would be trimmed in
        trim_votes = torch.zeros(len(inputs))
        
        # For each coordinate, identify clients that would be trimmed
        for coord_idx in range(stacked_flat.shape[1]):
            coord_values = stacked_flat[:, coord_idx]
            sorted_indices = torch.argsort(coord_values)
            
            # Mark clients that would be trimmed (first tm_b and last tm_b)
            trimmed_indices = torch.cat([sorted_indices[:tm_b], sorted_indices[-tm_b:]])
            for idx in trimmed_indices:
                trim_votes[idx] += 1
        
        # Select clients with fewer trim votes (those that are less likely to be outliers)
        # Sort by trim votes and select middle clients
        sorted_by_votes = torch.argsort(trim_votes)
        
        # Select middle clients (those with moderate trim votes)
        num_to_select = len(inputs) - 2 * tm_b
        selected_indices = sorted_by_votes[:num_to_select]
        
        return set(selected_indices.tolist())
    
    def _get_client_selection_trimmed_mean_old(self, inputs, step_index=0):
        """
        Get client selection from Trimmed Mean by identifying non-trimmed clients.
        This is an approximation since TM doesn't explicitly select clients.
        """
        # Get the aggregator instance to use its configured parameters
        aggr_name = self.aggregator_list[step_index]
        tm_aggregator = self.aggregators[aggr_name]
        tm_b = tm_aggregator.b
        
        # Compute distances from mean to identify potential outliers
        stacked = torch.stack(inputs, dim=0)
        mean_estimate = stacked.mean(dim=0)
        
        # Calculate distances from mean
        distances = torch.norm(stacked - mean_estimate.unsqueeze(0), dim=1)
        
        # Select clients that would not be trimmed (middle clients)
        sorted_indices = torch.argsort(distances)
        if tm_b is not None and tm_b > 0 and len(inputs) > 2 * tm_b:
            selected_indices = sorted_indices[tm_b:-tm_b]
        else:
            selected_indices = sorted_indices
        
        return set(selected_indices.tolist())
    
    def _get_client_selection_clipping(self, inputs, step_index=0):
        """
        Get client selection from Clipping by performing gradient sanitization.
        Unlike other aggregators that eliminate clients, clipping sanitizes gradients.
        Returns: (selected_indices, sanitized_inputs)
        """
        # Get reference point for clipping (last aggregated result or zero)
        if self.last_aggregated_result is not None:
            ref = self.last_aggregated_result
        else:
            ref = torch.zeros_like(inputs[0])
        
        # Perform gradient sanitization using clip_and_update
        sanitized_inputs = self.clip_and_update(inputs, ref)
        
        # Return all clients since clipping doesn't eliminate any clients
        # Also return sanitized inputs for the cascading process
        return set(range(len(inputs))), sanitized_inputs
    
    def _get_client_selection_rfa(self, inputs, step_index=0):
        """
        Get client selection from RFA by performing robust aggregation.
        Unlike other aggregators that eliminate clients, RFA processes all inputs robustly.
        Returns: (selected_indices, processed_inputs)
        """
        # Get the aggregator instance to use its configured parameters
        aggr_name = self.aggregator_list[step_index]
        rfa_aggregator = self.aggregators[aggr_name]
        
        # Apply RFA to get a robust aggregate
        robust_aggregate = rfa_aggregator(inputs)
        
        # For cascading, we can use the robust aggregate as a reference point
        # and create "processed" inputs that are closer to the robust center
        processed_inputs = []
        for inp in inputs:
            # Optionally apply some smoothing towards the robust center
            # For now, we keep the original inputs but could add smoothing here
            processed_inputs.append(inp)
        
        # Return all clients since RFA doesn't eliminate any clients
        # Also return processed inputs for the cascading process
        return set(range(len(inputs))), processed_inputs
    
    def _get_client_selections(self, inputs):
        """
        Get client selections from all aggregators in a cascading manner.
        Each aggregator operates on the filtered output of the previous one.
        """
        selections = {}
        current_inputs = inputs.copy()  # Make a copy to avoid modifying original
        current_client_indices = list(range(len(inputs)))  # Track original client indices
        
        # print(f"[Hybrid] Starting cascading selection with {len(inputs)} clients")
        
        for step_index, aggr_name in enumerate(self.aggregator_list):
            # print(f"[Hybrid] Step {step_index + 1}: {aggr_name}")
            # print(f"[Hybrid]   Input: {len(current_inputs)} clients (indices: {current_client_indices})")
            
            if aggr_name.lower() in ['multi_krum', 'krum']:
                selected_relative = self._get_client_selection_multikrum(current_inputs, step_index)
                
            elif aggr_name.lower() in ['trimmed_mean', 'tm']:
                selected_relative = self._get_client_selection_trimmed_mean(current_inputs, step_index)
                
            elif aggr_name.lower() in ['clipping', 'centered_clipping', 'cc']:
                # For clipping, we get both selection and sanitized inputs
                selected_relative, sanitized_inputs = self._get_client_selection_clipping(current_inputs, step_index)
                current_inputs = sanitized_inputs  # Use sanitized inputs for next step
                
            elif aggr_name.lower() in ['rfa', 'robust_federated_averaging']:
                # For RFA, we get both selection and processed inputs
                selected_relative, processed_inputs = self._get_client_selection_rfa(current_inputs, step_index)
                current_inputs = processed_inputs  # Use processed inputs for next step
                
            else:
                raise ValueError(f"Unknown aggregator for selection: {aggr_name}")
            
            # Convert relative indices back to original client indices
            selected_original = set(current_client_indices[i] for i in selected_relative)
            selections[aggr_name] = selected_original
            
            # Filter inputs and update indices for next step (except for clipping and RFA which already updated inputs)
            if aggr_name.lower() not in ['clipping', 'centered_clipping', 'cc', 'rfa', 'robust_federated_averaging']:
                selected_list = sorted(list(selected_relative))
                current_inputs = [current_inputs[i] for i in selected_list]
                current_client_indices = [current_client_indices[i] for i in selected_list]
            
            # print(f"[Hybrid]   Selected: {len(selected_original)} clients (original indices: {sorted(selected_original)})")
            # print(f"[Hybrid]   Output: {len(current_inputs)} clients for next step")
        
        # print(f"[Hybrid] Final selection: {len(current_inputs)} clients")
        # Return both selections and the final processed inputs
        return selections, current_inputs
    
    def _compute_final_selection(self, selections):
        """
        Compute final selection after cascading filtering.
        With cascading approach, the final selection is the output of the last aggregator.
        """
        if not selections:
            return set(range(self.n))
        
        # In cascading mode, the final selection is from the last aggregator
        final_aggregator = self.aggregator_list[-1]
        final_selection = selections[final_aggregator]
        
        # Ensure we have at least one client
        if len(final_selection) == 0:
            # print("[Hybrid] Warning: No clients selected, falling back to all clients")
            final_selection = set(range(self.n))
        
        return final_selection
    
    def __call__(self, inputs):
        """
        Perform hybrid aggregation.
        
        Args:
            inputs (list of torch.Tensor): List of client updates/gradients
            
        Returns:
            torch.Tensor: Aggregated result
        """
        if len(inputs) != self.n:
            raise ValueError(f"Expected {self.n} inputs, got {len(inputs)}")
        
        # Step 1: Get client selections and final processed inputs from cascading
        selections, final_inputs = self._get_client_selections(inputs)
        
        # Step 2: Compute final selection from cascading filtering
        final_selection = self._compute_final_selection(selections)
        
        # Update statistics
        self.elimination_stats = {
            f"{aggr}_selected": len(sel) for aggr, sel in selections.items()
        }
        self.final_selection_stats = {
            'final_size': len(final_selection),
            'selected_clients': sorted(list(final_selection))
        }
        
        # Track Byzantine bypass
        byzantine_clients = set(range(self.n - self.f, self.n))  # Assume last f clients are Byzantine
        bypassed = len(final_selection.intersection(byzantine_clients))
        self.bypass_stats = {
            'byzantine_bypassed': bypassed,
            'bypass_rate': bypassed / self.f if self.f > 0 else 0
        }
        
        # Step 3: Apply final aggregator on processed inputs
        if len(final_inputs) == 0:
            # Fallback: use mean of all inputs
            return torch.stack(inputs, dim=0).mean(dim=0)
        
        # Use the last aggregator for final aggregation on the processed inputs
        final_aggregator_name = self.aggregator_list[-1]
        
        if final_aggregator_name.lower() in ['multi_krum', 'krum']:
            # For Krum, just return mean of processed inputs
            result = torch.stack(final_inputs, dim=0).mean(dim=0)
            
        elif final_aggregator_name.lower() in ['trimmed_mean', 'tm']:
            # Apply coordinate-wise trimmed mean on processed inputs
            final_aggregator = self.aggregators[final_aggregator_name]
            result = final_aggregator(final_inputs)
            
        elif final_aggregator_name.lower() in ['clipping', 'centered_clipping', 'cc']:
            # For clipping, the inputs are already sanitized, just return their mean
            result = torch.stack(final_inputs, dim=0).mean(dim=0)
            
        elif final_aggregator_name.lower() in ['rfa', 'robust_federated_averaging']:
            # For RFA, apply the robust aggregation to processed inputs
            final_aggregator = self.aggregators[final_aggregator_name]
            result = final_aggregator(final_inputs)
        
        else:
            # Fallback: simple mean of processed inputs
            result = torch.stack(final_inputs, dim=0).mean(dim=0)
        
        # Update last aggregated result and iteration count
        self.last_aggregated_result = result.clone().detach()
        self.iteration_count += 1
        
        return result
    
    def get_attack_stats(self):
        """Return comprehensive attack statistics."""
        stats = {}
        
        # Add elimination statistics
        stats.update(self.elimination_stats)
        
        # Add final selection statistics
        stats.update(self.final_selection_stats)
        
        # Add bypass statistics
        stats.update(self.bypass_stats)
        
        # Add individual aggregator statistics
        for aggr_name, aggregator in self.aggregators.items():
            if hasattr(aggregator, 'get_attack_stats'):
                try:
                    aggr_stats = aggregator.get_attack_stats()
                    if aggr_stats:  # Only add if not None/empty
                        for key, value in aggr_stats.items():
                            stats[f"{aggr_name}_{key}"] = value
                except (AttributeError, KeyError) as e:
                    # Silently skip aggregators with missing attributes
                    pass
        
        return None
    
    def get_selection_details(self):
        """Return detailed information about client selections."""
        return {
            'elimination_stats': self.elimination_stats,
            'final_selection_stats': self.final_selection_stats,
            'bypass_stats': self.bypass_stats
        }


class AdaptiveHybridAggregator(_BaseAggregator):
    """
    Adaptive version of hybrid aggregator that can adjust aggregator weights
    based on their performance over time.
    """
    
    def __init__(self, n, m, tau, aggregator_list, adaptation_window=10):
        super(AdaptiveHybridAggregator, self).__init__()
        self.base_hybrid = HybridAggregator(n, m, tau, aggregator_list)
        self.adaptation_window = adaptation_window
        
        # Performance tracking
        self.performance_history = {aggr: [] for aggr in aggregator_list}
        self.aggregator_weights = {aggr: 1.0 for aggr in aggregator_list}
        
    def _update_performance(self, selections, ground_truth_malicious=None):
        """Update performance metrics for each aggregator."""
        if ground_truth_malicious is None:
            # Assume last f clients are malicious
            ground_truth_malicious = set(range(self.base_hybrid.n - self.base_hybrid.f, 
                                             self.base_hybrid.n))
        
        for aggr_name, selection in selections.items():
            # Calculate precision: how many selected are actually benign
            benign_selected = len(selection - ground_truth_malicious)
            precision = benign_selected / len(selection) if len(selection) > 0 else 0
            
            # Store performance
            self.performance_history[aggr_name].append(precision)
            
            # Keep only recent history
            if len(self.performance_history[aggr_name]) > self.adaptation_window:
                self.performance_history[aggr_name].pop(0)
            
            # Update weights based on recent performance
            if len(self.performance_history[aggr_name]) >= 3:
                avg_performance = np.mean(self.performance_history[aggr_name])
                self.aggregator_weights[aggr_name] = max(0.1, float(avg_performance))
    
    def __call__(self, inputs, ground_truth_malicious=None):
        """Perform adaptive hybrid aggregation."""
        # Get selections from base hybrid
        selections, final_inputs = self.base_hybrid._get_client_selections(inputs)
        
        # Update performance
        self._update_performance(selections, ground_truth_malicious)
        
        # Weighted intersection based on aggregator performance
        weighted_selections = {}
        for aggr_name, selection in selections.items():
            weight = self.aggregator_weights[aggr_name]
            weighted_selections[aggr_name] = (selection, weight)
        
        # Compute weighted consensus
        client_scores = {}
        for client_id in range(self.base_hybrid.n):
            score = 0
            total_weight = 0
            for aggr_name, (selection, weight) in weighted_selections.items():
                if client_id in selection:
                    score += weight
                total_weight += weight
            client_scores[client_id] = score / total_weight if total_weight > 0 else 0
        
        # Select top clients based on weighted scores
        min_clients = max(1, self.base_hybrid.n - 2 * self.base_hybrid.f)
        sorted_clients = sorted(client_scores.items(), key=lambda x: x[1], reverse=True)
        final_selection = set([client for client, _ in sorted_clients[:min_clients]])
        
        # Update base hybrid's selection stats
        self.base_hybrid.final_selection_stats = {
            'intersection_size': len(final_selection),
            'selected_clients': sorted(list(final_selection)),
            'aggregator_weights': self.aggregator_weights.copy()
        }
        
        # Perform final aggregation
        selected_inputs = [inputs[i] for i in sorted(final_selection)]
        final_aggregator_name = self.base_hybrid.aggregator_list[-1]
        
        if final_aggregator_name.lower() in ['multi_krum', 'krum']:
            result = torch.stack(selected_inputs, dim=0).mean(dim=0)
        elif final_aggregator_name.lower() in ['trimmed_mean', 'tm']:
            tm_aggregator = self.base_hybrid.aggregators[final_aggregator_name]
            result = tm_aggregator(selected_inputs)
        elif final_aggregator_name.lower() in ['clipping', 'centered_clipping']:
            clipping_aggregator = self.base_hybrid.aggregators[final_aggregator_name]
            
            # Set the momentum for clipping based on iteration count
            if self.base_hybrid.iteration_count == 0 or self.base_hybrid.last_aggregated_result is None:
                clipping_aggregator.momentum = torch.zeros_like(selected_inputs[0])
            else:
                clipping_aggregator.momentum = self.base_hybrid.last_aggregated_result.clone()
            
            result = clipping_aggregator(selected_inputs)
        else:
            result = torch.stack(selected_inputs, dim=0).mean(dim=0)
        
        # Update base hybrid's last aggregated result and iteration count
        self.base_hybrid.last_aggregated_result = result.clone().detach()
        self.base_hybrid.iteration_count += 1
        
        return result
    
    def get_attack_stats(self):
        """Return comprehensive statistics including adaptation info."""
        stats = self.base_hybrid.get_attack_stats()
        stats.update({
            'aggregator_weights': self.aggregator_weights,
            'avg_performance': {aggr: np.mean(hist) if hist else 0 
                              for aggr, hist in self.performance_history.items()}
        })
        return stats
