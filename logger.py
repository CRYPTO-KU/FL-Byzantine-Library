import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from model_registry import get_net
from utils import  get_layer_dims

def save_results(args, sparse_mask, avg_attack_stats=None, **kwargs):
    save_loc_given = args.save_loc
    sim_id = random.randint(1,99999)
    attack = args.attack
    if attack == 'sparse':
       attack+='_Cfg{}'.format(args.sparse_cfg)
    aggr = args.aggr
    if aggr =='cc':
        aggr = '{}_Tau_{}'.format(aggr,args.tau)
    momentum = args.Lmomentum
    dataset = '{}_{}'.format(args.dataset_name,args.dataset_dist)
    path_ = 'ATK_{}-Def_{}-dist_{}-{}'.format(attack,aggr,
                                                    dataset,sim_id)
    path = 'Results'
    if args.aggr =='avg' and args.traitor ==0:
        path_ = 'Baseline-{}-B_{}-{}'.format(dataset,momentum,sim_id)
    elif args.traitor == 0:
        path_ = '{}-No_Attacker-{}-B_{}-{}'.format(aggr,dataset, momentum, sim_id)
    if not os.path.exists(path):
        os.mkdir(path)
    if save_loc_given != '':
        path = os.path.join(path,save_loc_given)
        if not os.path.exists(path):
            os.mkdir(path)
    path = os.path.join(path,path_)
    os.mkdir(path)
    log(path,args,sim_id)
    
    
    vec_path = os.path.join(path,'vecs')
    std_path = os.path.join(path, 'std')
    os.mkdir(vec_path)
    os.mkdir(std_path)
    if sparse_mask is not None:
        file_name = 'PruneMask-Plot.png'
        mask_name = 'Best_mask'
        net = get_net(args)
        save = os.path.join(path, file_name)
        pruned_histogram(save,net,sparse_mask)
        mask_file = os.path.join(path,mask_name)
        mask_inds = torch.arange(0,sparse_mask.numel(),device=sparse_mask.device)
        mask_inds = mask_inds[sparse_mask>0]
        torch.save(mask_inds.detach().cpu(), mask_file)
    for key,vals in kwargs.items():
        if vals is not None:
            x_ = list(range(1,args.global_epoch+1))
            if isinstance(vals,int):
                mean_val, std = -1,-1
            else:
                mean_val, std = vals.mean(axis=0),vals.std(axis=0)
            std = np.around(std,decimals=3)
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
                    'STD  :   {}'.format(key,mean_val[-1],(std[-1])) + '\n')
            f.close()
            np_file = os.path.join(vec_path,key)
            np_file_std = os.path.join(std_path,key)
            np.save(np_file,mean_val)
            np.save(np_file_std,std)
    if avg_attack_stats:
    # Also append to main log
        with open(path + '/log.txt', 'a') as f:
            f.write('\n############## Attack Statistics ###############\n')
            for stat_name, avg_value in avg_attack_stats.items():
                f.write(f'{stat_name}: {avg_value:.4f}\n')
            f.write('############## End Attack Statistics ###############\n')
    return

def log(path,args,sim_id):
    n_path = path
    f = open(n_path + '/log.txt', 'w+')
    f.write('############## Args ###############' + '\n')
    l =  'sim_id : {} \n'.format(sim_id)
    f.write(l)
    for arg in vars(args):
        line = str(arg) + ' : ' + str(getattr(args, arg))
        f.write(line + '\n')
    f.write('############ Results ###############' + '\n')
    f.close()


def pruned_histogram(loc,net,mask,exclude_bns = True):
    dims, flat_dims,names,colors = get_layer_dims(net)
    cum_dim = [0]
    cum_dim.extend(flat_dims)
    cum_dim = np.cumsum(cum_dim)
    mask = mask.detach().cpu().numpy()
    totals = []
    total_max = 0
    for i in range(len(cum_dim)-1):
        b,e = cum_dim[i],cum_dim[i+1]
        p = mask[b:e]
        size = len(p)
        pruned = np.sum(p)
        total = round((pruned/size) *100,1)
        totals.append(total)
        if 99 > total > total_max:
            total_max = total
    if exclude_bns:
        colors = np.asarray(colors)
        weights = colors != 'tab:red'
        colors = colors[weights]
        totals = np.asarray(totals)[weights]
        names = np.asarray(names)[weights]
    fig, ax = plt.subplots(figsize=(8,3))
    ax.bar(names,totals,color=colors)
    ax.set_ylabel('Left parameters %')
    ax.set_title('Pruning weights')
    ax.set_xlabel('Layers')
    ax.set_ylim([0,total_max+5])
    plt.tight_layout()
    plt.savefig(loc)
    return