import argparse
import math

def args_parser():
    parser = argparse.ArgumentParser()
    # technical params
    parser.add_argument('--trials', type=int, default=1, help='number of trials')
    parser.add_argument('--gpu_id', type=int, default=0, help='Use cuda as device')

    # Federated params
    parser.add_argument('--global_epoch', type=int, default=100, help='total cumulative epoch')
    parser.add_argument('--localIter', type=int, default=1, help='Local Epoch')
    parser.add_argument('--num_client', type=int, default=25, help='number of clients')
    parser.add_argument('--traitor', type=float, default=0.2, help='traitor ratio')
    parser.add_argument('--attack', type=str, default='rop', help='see attacks_deprecated.py')
    parser.add_argument('--aggr', type=str, default='gas-bulyan', help='robust Aggregators')
    parser.add_argument('--embd_momentum', type=bool, default=False, help='FedADC embedded momentum')
    parser.add_argument('--early_stop', type=bool, default=False, help='Early stop function')
    parser.add_argument('--MITM', type=bool, default=True, help='Adversary capable of man-in-middle-attack')

    # Defence params
    parser.add_argument('--tau', type=float, default=1, help='Radius of the ball for CC aggregator')
    parser.add_argument('--n_iter', type=int, default=1, help='number of iteration for cc aggr')
    parser.add_argument('--buck_len', type=int, default=3, help='bucket length for sequential cc')
    parser.add_argument('--buck_avg', type=bool, default=True, help='average the bucket for sequential cc')
    parser.add_argument('--multi_clip', type=bool, default=False, help='average the bucket for sequential cc')
    parser.add_argument('--T', type=int, default=3, help='RFA inner iteration')
    parser.add_argument('--nu', type=float, default=0.1, help='RFA norm budget')
    parser.add_argument('--gas_p', type=int, default=1000, help='Number of chunks for GAS')

    # attack params
    parser.add_argument('--z_max', type=float, default=None, help='attack scale,none for auto generate')
    parser.add_argument('--alie_z_max', type=list, default=1.2, help='attack scale for ALIE attack,none for auto generate')
    parser.add_argument('--nestrov_attack', type=bool, default=False, help='clean step first- For non-omniscient attacks')
    parser.add_argument('--epsilon', type=float, default=0.2, help='IPM attack scale')
    parser.add_argument('--pert_vec', type=str, default='std', help='[unit_vec,sign,std] for Minmax and Minsum attacks')
    # modular design for ROP attack
    parser.add_argument('--pi', type=float, default=0, help='location of the attack,1 for full relocation to aggr reference')
    parser.add_argument('--angle', type=int, default=90, help='angle of the pert, 180,90 and none')
    parser.add_argument('--lamb', type=float, default=0.9, help='refence point for attack if angle is not none')


    # optimiser related
    parser.add_argument('--opt', type=str, default='sgd', help='name of the optimiser')
    parser.add_argument('--lr', type=float, default=0.1, help='learning_rate')
    parser.add_argument('--lr_decay', type=list, default=[75], help='lr drop at the given epochs')
    parser.add_argument('--wd', type=float, default=0, help='weight decay Value')
    parser.add_argument('--Lmomentum', type=float, default=.9, help='Local Momentum for SGD')
    parser.add_argument('--betas', type=tuple, default=(0.9,0.999), help='betas for adam and adamw opts')
    parser.add_argument('--worker_momentum', type=bool, default=True, help='adam like gradiant multiplier for SGD (1-Lmomentum)')
    parser.add_argument('--nesterov', type=bool, default=False, help='nestrov momentum for Local SGD steps')

    # dataset related
    parser.add_argument('--dataset_name', type=str, default='cifar10', help='see data_loader.py')
    parser.add_argument('--dataset_dist', type=str, default='iid',
                        help='distribution of dataset; iid or sort_part, dirichlet')
    parser.add_argument('--numb_cls_usr', type=int, default=2,
                        help='number of label type per client if sort_part selected')
    parser.add_argument('--alpha', type=float, default=1.,
                        help='alpha constant for dirichlet dataset_dist,lower for more skewness')
    parser.add_argument('--bs', type=int, default=32, help='batchsize')

    # nn related
    parser.add_argument('--nn_name', type=str, default='resnet20', help='simplecnn,simplecifar,VGGs resnet(8-9-18-20)')
    parser.add_argument('--weight_init', type=str, default='-',
                        help='nn weight init, kn (Kaiming normal) or - (None)')
    parser.add_argument('--norm_type', type=str, default='bn',
                        help='gn (GroupNorm), bn (BatchNorm), - (None)')
    parser.add_argument('--num_groups', type=int, default=32,
                        help='number of groups if GroupNorm selected as norm_type, 1 for LayerNorm')

    # sparse related
    parser.add_argument("--sparse_mask_path", type=str, default=None,
                        help='Load sparse mask from given path')

    args = parser.parse_args()
    return args
