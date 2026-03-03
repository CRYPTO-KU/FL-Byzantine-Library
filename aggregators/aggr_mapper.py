from .clipping import Clipping
from .cm import CM
from .krum import Krum
from .rfa import RFA
from .trimmed_mean import TM
from .fedavg import fedAVG
from .cc_seq import Clipping_seq
from .bulyan import Bulyan
from .sign_sgd import SignSGD
from .gas import Gas
from .fl_trust import FL_trust
from .cc_seq_ecc import Clipping_seq_ecc
from .tm_history import TM_history
from .tm_perfect import TM_perfect
from .cc_cluster import Clipping_cluster
from .Bucketing import Bucketing
from .hybrid_aggr import HybridAggregator, AdaptiveHybridAggregator
from .hybrid_aggr2 import HybridAggregator as HybridAggregator2
from .foolsGold import FoolsGold
from .scc_krum import Clipping_seq_krum
from .tm_cheby import ChebyshevAggregator
from .tm_capped import TM_capped
from .tm_abs import TM_Abs
from .med_krum import MedKrum
from .foundation import FoundationFL, FoundationFL_TrimmedMean, FoundationFL_Median
from .lasa import LASA
from .signguard import SignGuard
from .fedredefense import FedREDefense
from .fl_defender import FLDefender
from .skymask import SkyMask
from .fldetector import FLDetector
from .flame import Flame
from .dnc import DnC
from .fedseca import FedSECA


aggr_mapper = {'cc': Clipping, 'cm': CM, 'krum': Krum, 'rfa': RFA, 'tm': TM,'avg':fedAVG,
               'ccs':Clipping_seq,'scc':Clipping_seq,'bulyan':Bulyan,'sign':SignSGD, 'ccs_ecc':Clipping_seq_ecc,
               'gas':Gas,'fl_trust':FL_trust,
               'tm_perfect':TM_perfect,'tm_history':TM_history,
               'cc_cluster':Clipping_cluster,
               'hybrid': HybridAggregator, 'adaptive_hybrid': AdaptiveHybridAggregator,
               'hybrid2': HybridAggregator2, 'foolsgold': FoolsGold,
               'scc_krum': Clipping_seq_krum, 'tm_cheby': ChebyshevAggregator,
               'tm_capped': TM_capped,'tm_abs': TM_Abs,
               'med_krum': MedKrum, 'foundation': FoundationFL, 
               'foundation_tm': FoundationFL_TrimmedMean, 'foundation_med': FoundationFL_Median,
                'lasa': LASA, 'signguard': SignGuard,
                'fedredefense': FedREDefense, 'fl_defender': FLDefender,
                'skymask': SkyMask, 'fldetector': FLDetector, 'flame': Flame,
                'dnc': DnC, 'fedseca': FedSECA}

def get_bucketing(args):
    bucketing = Bucketing(args.buck_len, args.bucket_type, args.bucket_op)
    return bucketing

def get_aggr(args,net_ps,device,num_client,num_traitor,**kwargs):
    alg = args.aggr
    if '-' in alg:
        alg,secondry_alg = alg.split('-')
    params = set_aggr_params(args, num_client, num_traitor, net_ps, device,**kwargs)
    return aggr_mapper[alg](**params[alg])

def set_aggr_params(args,num_client,b,net_ps,device,**kwargs):
    secondry_alg = None
    if '-' in args.aggr:
        alg,secondry_alg = args.aggr.split('-')
    hybrid_aggr_list = args.hybrid_aggregator_list.split('+') if args.hybrid_aggregator_list else []
    # Handle tau parameter - extract scalar value if it's a list (for grid search compatibility)
    tau_val = args.tau[0] if isinstance(args.tau, list) else args.tau
    print('num_byzatine clients: ', b)
    if args.bucketing:
        n= num_client-b
    else:
        n= num_client-b-2
    aggr_params = {
    'krum': {'n': num_client, 'f': b, 'm': n},
    'cc': {'tau': tau_val,'b':b},
    'ccs': {'tau': tau_val, 'buck_len': args.buck_len, 'buck_avg': args.buck_avg},
    'ccs_ecc': {'args': args},
    'cm': {},
    'sign': {'args': args},
    'rfa': {'T': args.T, 'nu': args.nu},
    'tm': {'b': b},
    'avg': {},
    'bulyan': {'n': num_client, 'm': b},
    'gas': {'n': num_client, 'm': b, 'p': args.gas_p, 'aggr': secondry_alg},
    'fl_trust': {'root_dataset': kwargs.get('root_dataset'), 'model': net_ps, 'args': args, 'device': device},
    'tm_perfect': {'b': b},
    'tm_history': {'b': b},
    'cc_cluster': {'tau': tau_val, 'buck_len': args.buck_len, 'buck_avg': args.buck_avg,
                   'num_clustering': args.num_clustering, 'bucket_shift': args.bucket_shift,
                   'shift_amount': args.shift_amount,'b':b},
    'hybrid': {'n': num_client, 'm': b, 'f': n, 'tau': tau_val, 'aggregator_list': hybrid_aggr_list},
    'hybrid2': {'n': num_client, 'm': b, 'tau': tau_val, 'aggregator_list': hybrid_aggr_list},
    'foolsgold': {'n': num_client, 'use_memory': getattr(args, 'fg_use_memory', True), 
                  'memory_size': getattr(args, 'fg_memory_size', 10), 
                  'epsilon': getattr(args, 'fg_epsilon', 1e-5)},
    'scc_krum': {'tau': tau_val, 'm': b, 'buck_len': args.buck_len, 'buck_avg': args.buck_avg,
                 'bucket_op': args.bucket_op},
    'tm_cheby': {'n': num_client, 'k_sigma': getattr(args, 'cheby_k_sigma')},
    'tm_capped': {'n': num_client, 'm': b},
    'tm_abs': {'b': b},
    'med_krum': {'n': num_client, 'f': b, 'm': n},
    'foundation': {'b': b, 'num_synthetic': getattr(args, 'foundation_num_synthetic', 2)},
    'foundation_tm': {'b': b, 'num_synthetic': getattr(args, 'foundation_num_synthetic', 2)},
    'foundation_med': {'num_synthetic': getattr(args, 'foundation_num_synthetic', 2)},
    'lasa': {'layer_dims': kwargs.get('lasa_dims'), 'num_clients': num_client,
                      'sparsity_ratio': getattr(args, 'lasa_sparsity_ratio', 0.7),
                      'lambda_n': getattr(args, 'lasa_lambda_n', 1.0),
                      'lambda_s': getattr(args, 'lasa_lambda_s', 1.0)},
    'signguard': {'num_clients': num_client,
                  'sparsity_ratio': getattr(args, 'signguard_sparsity_ratio', 0.1),
                  'norm_bounds': (getattr(args, 'signguard_norm_lower', 0.1), 
                                 getattr(args, 'signguard_norm_upper', 3.0)),
                  'clustering_method': getattr(args, 'signguard_clustering', 'meanshift'),
                  'iterations': getattr(args, 'signguard_iterations', 1),
                  'eps': getattr(args, 'signguard_eps', 0.05),
                  'min_samples': getattr(args, 'signguard_min_samples', 2)},
    'fedseca': {'sparsity_gamma': getattr(args, 'fedseca_sparsity_gamma', 0.9)},
    'fedredefense': {'n_clients': num_client,
                     'n_components': getattr(args, 'fedredefense_n_components', 3),
                     'threshold': getattr(args, 'fedredefense_threshold', 0.6)},
    'fl_defender': {'n_clients': num_client},
    'skymask': {'root_dataset': kwargs.get('root_dataset'), 'net_ps': net_ps,
                'args': args, 'device': device,
                'mask_lr': getattr(args, 'skymask_lr', 0.01),
                'mask_epochs': getattr(args, 'skymask_epochs', 20)},
    'fldetector': {'n_clients': num_client,
                   'warmup_rounds': getattr(args, 'fldetector_warmup', 10),
                   'lbfgs_history': getattr(args, 'fldetector_lbfgs_history', 5)},
    'flame': {'epsilon': getattr(args, 'flame_epsilon', 3000),
              'delta': getattr(args, 'flame_delta', 0.01),
              'add_noise': getattr(args, 'flame_add_noise', True)},
    'dnc': {'num_clients': num_client, 'num_byzantine': b,
            'sub_dim': getattr(args, 'dnc_sub_dim', 10000),
            'num_iters': getattr(args, 'dnc_num_iters', 5),
            'filter_frac': getattr(args, 'dnc_filter_frac', 1.0)}
    }
    return aggr_params
