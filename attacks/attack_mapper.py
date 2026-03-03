from .alie import alie
from .ipm import IPMAttack
from .rop import reloc
from .bit_flip import pgd_traitor
from .label_flip import label_flip_traitor
from .cw import cw_traitor
from .sparse import sparse
from .minmax import minmax
from .minsum import minsum

from .sparse_opted import sparse_optimized
from .fang import fang
from .local_model_poisoning import (
    KrumAttack, TrimmedMeanAttack, EdgeCaseAttack, LocalMinMaxAttack, 
    AdaptiveKrumAttack, LocalTrimmedMeanAttack, StealthyAttack
)
from .mimic import (
    MimicAttack, MimicVariantAttack, AdaptiveMimicAttack
)
from .lasa_attack import lasa_attack

attack_mapper ={'bit_flip':pgd_traitor,'label_flip':label_flip_traitor,
                'cw':cw_traitor,'alie':alie,'reloc':reloc,'fang':fang,
                'ipm':IPMAttack,'sparse':sparse,'minmax':minmax,'minsum':minsum
                ,'sparse_opt':sparse_optimized,
                'krum_attack':KrumAttack,'trimmed_mean_attack':TrimmedMeanAttack,
                'edge_case':EdgeCaseAttack,'local_minmax':LocalMinMaxAttack,
                'adaptive_krum':AdaptiveKrumAttack,'local_trimmed_mean':LocalTrimmedMeanAttack,
                'stealthy':StealthyAttack,
                'mimic':MimicAttack,'mimic_variant':MimicVariantAttack,
                'adaptive_mimic':AdaptiveMimicAttack,'lasa':lasa_attack}


def get_attacker_client(id,dataset,device,args,layer_inds):
    num_client = args.num_client
    num_traitor = int(args.traitor*num_client) if args.traitor < 1 else int(args.traitor)
    client_params = {'id':id,'dataset':dataset,'device':device,'args':args}
    attacker_params = {'n':num_client,'m':num_traitor,'z':args.z_max,'eps':args.epsilon,'layer_inds':layer_inds}
    mal_client = attack_mapper[args.attack](**attacker_params,**client_params)
    return mal_client