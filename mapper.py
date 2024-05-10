from Aggregators.clipping import Clipping
from Aggregators.cm import CM
from Aggregators.krum import Krum
from Aggregators.rfa import RFA
from Aggregators.trimmed_mean import TM
from Aggregators.fedavg import fedAVG
from Aggregators.cc_seq import Clipping_seq
from Aggregators.bulyan import Bulyan
from Aggregators.sign_sgd import SignSGD
from Aggregators.gas import Gas
from Aggregators.fl_trust import FL_trust
from Attacks.alie import alie
from Attacks.ipm import IPMAttack
from Attacks.rop import reloc
from Attacks.bit_flip import bit_flip_traitor
from Attacks.label_flip import label_flip_traitor
from Attacks.gaussian_noise import gaussian_noise_traitor
from Attacks.cw import cw_traitor
from Attacks.sparse import Sparse
from Attacks.minmax import minmax
from Attacks.minsum import minsum
from Attacks.sparse_opted import sparse_optimized
from client import client as loyal_client
import numpy as np
from data_loader import DatasetSplit


aggr_mapper = {'cc': Clipping, 'cm': CM, 'krum': Krum, 'rfa': RFA, 'tm': TM,'avg':fedAVG,
               'ccs':Clipping_seq,'bulyan':Bulyan,'sign':SignSGD,
               'gas':Gas,'fl_trust':FL_trust}
attack_mapper ={'bit_flip':bit_flip_traitor,'gaussian':gaussian_noise_traitor,'label_flip':label_flip_traitor,
                'cw':cw_traitor,'alie':alie,'reloc':reloc,'rop':reloc,
                'ipm':IPMAttack,'sparse':Sparse,'minmax':minmax,'minsum':minsum
                ,'sparse_opt':sparse_optimized}



def get_aggr(args,test_set,net_ps,device):
    alg = args.aggr
    secondry_alg = None
    if '-' in alg:
        alg,secondry_alg = alg.split('-')
    num_client = args.num_client
    b= int(num_client * args.traitor)
    n= num_client-b-2
    p = args.gas_p
    root_dataset = None
    if alg == 'fl_trust':
        l = test_set.__len__()
        root_dataset_inds = np.random.choice(range(l),100, replace=False)
        root_dataset = DatasetSplit(test_set,root_dataset_inds)
    aggr_params = {'cc': [args.tau], 'ccs':[args.tau,args.buck_len,args.buck_rand,args.buck_avg]
        ,'cm': [None],'sign': [None],'krum': [num_client,b,n], 'rfa': [args.T,args.nu], 'tm': [b],'avg':[None],'bulyan':[num_client,b],
                   'gas':[num_client,b,n,p,secondry_alg],'fl_trust':[root_dataset,net_ps,args,device]}
    return aggr_mapper[alg](*aggr_params[alg])


def get_attacker_cl(id,dataset,device,args,prune_mask):
    num_client = args.num_client
    num_traitor = int(args.traitor*num_client) if args.traitor < 1 else int(args.traitor)
    client_params = {'id':id,'dataset':dataset,'device':device,'args':args}
    attacker_params = {'n':num_client,'m':num_traitor,'z':args.z_max,'eps':args.epsilon,'mask':prune_mask}
    traitor_client = attack_mapper[args.attack](**attacker_params,**client_params)
    return traitor_client

def get_benign_cl(id,dataset,device,args):
    client_params = {'id': id, 'dataset': dataset, 'device': device, 'args': args}
    return loyal_client(**client_params)