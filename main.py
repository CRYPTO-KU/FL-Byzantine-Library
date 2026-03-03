import torch
from fl import FL
from utils import *
import numpy as np
from logger import *
import time
from tqdm import tqdm
from mapper import Mapper
from config import args_parser
import os


def train_epoch(fl: FL):
    """Train for one epoch and collect metrics."""
    target_epoch = int(fl.epoch) + 1  # Target the next epoch
    metrics = {
        'losses': [],
        'aggr_times': [],
        'attack_stats': {}
    }
    
    while int(fl.epoch) < target_epoch:
        # Training step
        outputs = fl.train()
        metrics['losses'].append(fl.avg_train_loss)
        
        # Aggregation step with timing
        start_time = time.time()
        fl.aggregate(outputs)
        aggr_time = time.time() - start_time
        metrics['aggr_times'].append(aggr_time)
        
        # Update model
        fl.update_global_model()
        
        # Collect attack statistics
        attack_info = fl.get_aggr_success_info()
        if attack_info is not None and isinstance(attack_info, dict):
            for key, value in attack_info.items():
                if key in metrics['attack_stats']:
                    metrics['attack_stats'][key].append(value)
                else:
                    metrics['attack_stats'][key] = [value]
    
    # Calculate averages and test accuracy and return as dictionary
    test_acc = fl.evaluate_accuracy()

    epoch_summary = {
        'avg_loss': round(sum(metrics['losses']) / len(metrics['losses']), 4) if metrics['losses'] else 0.0,
        'avg_aggr_time': round(sum(metrics['aggr_times']) / len(metrics['aggr_times']), 4) if metrics['aggr_times'] else 0.0,
        'attack_stats': {k: round(sum(v) / len(v), 4) if v else 0.0 for k, v in metrics['attack_stats'].items()},
        'test_accuracy': test_acc * 100 if test_acc is not None else 0.0
    }
    
    return epoch_summary


def run(args, device):
    """Main training run with prettier logging."""
    if args.aggr == 'sign':
        args.lr = 0.01
    
    mapper = Mapper(args, device)
    fl = mapper.initialize_FL()
    
    # Initialize tracking lists
    accs, losses, aggr_times = [], [], []
    all_attack_stats = {}
    
    print(f"Starting training with {args.aggr} aggregator")
    print(f"Training for {args.global_epoch} epochs with {len(fl.benign_clients)} benign and {len(fl.malicious_clients)} malicious clients")
    print("-" * 60)
    
    for epoch in tqdm(range(args.global_epoch), desc="Training Progress"):
        # Train one epoch
        epoch_summary = train_epoch(fl)
        
        # Extract values from summary
        train_loss = epoch_summary['avg_loss']
        aggr_time = epoch_summary['avg_aggr_time']
        attack_stats = epoch_summary['attack_stats']
        test_acc = epoch_summary['test_accuracy']
        
        # Collect metrics
        losses.append(train_loss)
        aggr_times.append(aggr_time)
        accs.append(test_acc)

        
        # Aggregate attack statistics
        for key, value in attack_stats.items():
            if key in all_attack_stats:
                all_attack_stats[key].append(value)
            else:
                all_attack_stats[key] = [value]

        # Logging
        attack_info = ""
        if attack_stats:
            attack_summary = {k: f"{v:.3f}" for k, v in attack_stats.items()}
            attack_info = f" | Attack Stats: {attack_summary}"
        
        print(f"Epoch {epoch + 1:3d} | Acc: {test_acc :5.1f}% | Loss: {train_loss:.4f} | Aggr Time: {aggr_time:.4f}s{attack_info}")
        
        # Learning rate decay
        if epoch + 1 in args.lr_decay:
            fl.__update_lr__()

    # Final statistics
    print("-" * 80)
    print(f"Final accuracy: {accs[-1]:.1f}%")
    print(f"Average aggregation time: {np.mean(aggr_times):.4f}s")
    
    # Calculate average attack statistics
    avg_attack_stats = {}
    if all_attack_stats:
        print("Attack Statistics Summary:")
        for stat_name, stat_values in all_attack_stats.items():
            avg_stat = np.mean(stat_values)
            avg_attack_stats[stat_name] = avg_stat
            print(f"  {stat_name}: {avg_stat:.4f} (avg over {len(stat_values)} samples)")
    
    args.avg_aggr_time = sum(aggr_times) / len(aggr_times)
    final_mask = fl.final_prune_mask()
    try:
        print(sum(fl.malicious_clients[0].opted_z_vals) / len(fl.malicious_clients[0].opted_z_vals),'avg z val on',args.attack)
        z_vals = [v.item() if isinstance(v, torch.Tensor) else v for v in fl.malicious_clients[0].opted_z_vals]
        np.save(f"opted_z_vals-{args.aggr}-{args.sparse_th}.npy", z_vals)
    except Exception as e:
        pass
    return accs, losses, final_mask, avg_attack_stats


if __name__ == '__main__':
    args = args_parser()
    
    # Set device based on availability
    if torch.cuda.is_available():
        device = args.gpu_id if args.gpu_id > -1 else 'cpu'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    
    dims = (args.trials, args.global_epoch)
    accs_all, losses_all = np.empty(dims), np.empty(dims)
    last_accs = []
    prune_masks = []
    best_mask = None
    all_trials_attack_stats = {}
    
    for i in range(args.trials):
        accs, losses, prune_mask, trial_attack_stats = run(args, device)
        accs_all[i], losses_all[i] = accs, losses
        prune_masks.append(prune_mask)
        last_accs.append([accs[-1]])
        
        # Aggregate attack stats across trials
        if trial_attack_stats:
            for stat_name, stat_value in trial_attack_stats.items():
                if stat_name in all_trials_attack_stats:
                    all_trials_attack_stats[stat_name].append(stat_value)
                else:
                    all_trials_attack_stats[stat_name] = [stat_value]
    
    # Calculate average attack statistics across all trials
    avg_attack_stats = {}
    if all_trials_attack_stats:
        for stat_name, stat_values in all_trials_attack_stats.items():
            avg_attack_stats[stat_name] = np.mean(stat_values)
    
    if prune_masks[0] is not None:
        l = np.argmin(last_accs)
        best_mask = prune_masks[l]
    
    # Save results with attack statistics
    save_results(args, best_mask, avg_attack_stats, Test_acc=accs_all, Train_losses=losses_all)
