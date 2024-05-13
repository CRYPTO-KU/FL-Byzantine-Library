import torch
from parameters import args_parser
from nn_classes import get_net
from torch.utils.data import DataLoader
import data_loader as dl
from mapper import *
from utils import *
import time
import numpy as np

def run(args,device):
    num_client = args.num_client
    num_byz = int(args.traitor*num_client) if args.traitor < 1 else int(args.traitor)
    trainset, testset = dl.get_dataset(args)
    if args.traitor > 0:
        traitors = np.random.choice(range(num_client), num_byz, replace=False)
        if not args.MITM:
            sample_inds_byz, data_map_byz = dl.get_indices(trainset, args, test_set=None, num_cli_force=num_byz)
            byz_datasets = [dl.DatasetSplit(trainset, inds) for inds in sample_inds_byz.values()]
    else:
        traitors = []
    assert num_byz == len(traitors)
    loyal_clients,traitor_clients = [], []
    total_sample = trainset.__len__()
    sample_inds, data_map = dl.get_indices(trainset, args)
    net_ps = get_net(args).to(device) # global model (PS)
    print('number of parameters ', round(count_parameters(net_ps) / 1e6,3) ,'M')
    testloader = DataLoader(testset,128,shuffle=False,pin_memory=True)
    aggr = get_aggr(args,trainset,net_ps,device)
    epoch = 0
    lr = args.lr
    accs = []
    ep_loss, losses = [], []
    aggregation_times = []
    prune_mask = None
    sparse_attack = True if args.attack.split('_')[0] == 'sparse' else False
    if sparse_attack:
        prune_mask = load_sparse_mask(args,net_ps,device)
        print('Sparse mask ratio to network:', round((prune_mask.sum() / prune_mask.numel()).item(),3))
    for i in range(num_client):
        worker_dataset = dl.DatasetSplit(trainset,sample_inds[i])
        #worker_data_map = data_map[i]
        if i in traitors:
            traitor_clients.append(get_attacker_cl(i,worker_dataset,device,args,prune_mask))
            if len(traitor_clients) == 1:
                traitor_clients[0].debug_byz = True
        else:
            loyal_clients.append(get_benign_cl(i,worker_dataset,device,args))
    [cl.update_model(net_ps) for cl in [*loyal_clients, *traitor_clients]]
    if args.traitor > 0 and not args.MITM:
        [cl.update_dataloader(dataset) for cl,dataset in zip(traitor_clients,byz_datasets) if cl.omniscient]
    traitor_ms = []

    ## Start training
    while epoch < args.global_epoch:
        [cl.train_() for cl in loyal_clients]
        if num_byz >0:
            if traitor_clients[0].omniscient:
                if args.MITM:
                    benign_grads = [cl.get_grad() for cl in loyal_clients]
                else:
                    [cl.train_psuedo_moments() for cl in traitor_clients]
                    benign_grads = [cl.get_benign_preds() for cl in traitor_clients]
                    #benign_grads = functools.reduce(operator.iconcat, benign_grads, [])
                traitor_clients[0].omniscient_callback(benign_grads)
                traitor_ms = [traitor_clients[0].adv_momentum for cl in traitor_clients] # faster to do this than to call the function again
                #[cl.omniscient_callback(benign_grads) for cl in traitor_clients]
            else:
                [cl.train_() for cl in traitor_clients]
                traitor_ms = [cl.get_grad() for cl in traitor_clients]
        outputs = [cl.get_grad() for cl in loyal_clients]
        outputs.extend(traitor_ms)
        assert len(outputs) == num_client
        ep_loss.append(sum([cl.mean_loss for cl in loyal_clients]) / len(loyal_clients))
        t = time.time()
        robust_update = aggr.__call__(outputs)
        aggr_time = time.time() - t
        aggregation_times.append(aggr_time)
        ps_flat = get_model_flattened(net_ps, device)
        ps_flat.add_(robust_update, alpha=-lr)
        unflat_model(net_ps, ps_flat)
        prev_epoch = int(epoch)
        epoch += (num_client * args.localIter * args.bs) / total_sample
        current_epoch = int(epoch)
        [cl.update_model(net_ps) for cl in [*loyal_clients, *traitor_clients]]
        if num_byz > 0:
            if traitor_clients[0].relocate:
                [cl.get_global_m(robust_update.clone()) for cl in traitor_clients]
        if current_epoch > prev_epoch:
            acc = evaluate_accuracy(net_ps, testloader, device)
            mean_train_loss = round(sum(ep_loss)/len(ep_loss),4)
            print('Epoch',current_epoch,'Accuracy',round(acc,3) * 100,'|',
                  'Loss:',mean_train_loss,'|','Aggregation time:',round(sum(aggregation_times),3))
            accs.append(acc*100)
            losses.append(mean_train_loss)
            ep_loss = []
            if current_epoch in args.lr_decay:
                [cl.lr_step()for cl in [*loyal_clients,*traitor_clients]]
                lr *= .1
    return accs,losses


if __name__ == '__main__':
    args = args_parser()
    device = args.gpu_id if args.gpu_id > -1 else 'cpu'
    dims = (args.trials, args.global_epoch)
    accs_all, losses_all = np.empty(dims), np.empty(dims)
    bypass_all = np.empty(dims)
    tm_bypass_all = np.empty(dims)
    last_accs = []
    prune_masks = []
    for i in range(args.trials):
        accs, losses = run(args, device)
        last_accs.append([accs[-1]])
    save_results(args,Test_acc=accs_all, Train_losses=losses_all)
