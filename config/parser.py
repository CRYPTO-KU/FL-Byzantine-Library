"""
Composed configuration and CLI parser.

Provides:
- ``FLConfig``: a single dataclass that composes all sub-configs.
- ``parse_args()``:  parse CLI flags into an ``FLConfig``.
- ``args_parser()``: **legacy shim** that returns a flat ``argparse.Namespace``
  so that existing code continues to work without modification.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field, fields, asdict
from typing import Optional

from .base import ExperimentConfig
from .federation import FederationConfig
from .optimizer import OptimizerConfig
from .model import ModelConfig
from .defense import DefenseConfig
from .attack import AttackConfig
from .pruning import PruningConfig


# ── Composed config ──────────────────────────────────────────────────────────

@dataclass
class FLConfig:
    """Complete FL experiment configuration (composed of sub-configs)."""

    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    federation: FederationConfig = field(default_factory=FederationConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    defense: DefenseConfig = field(default_factory=DefenseConfig)
    attack: AttackConfig = field(default_factory=AttackConfig)
    pruning: PruningConfig = field(default_factory=PruningConfig)

    def to_flat_namespace(self) -> argparse.Namespace:
        """Flatten all sub-configs into a single ``argparse.Namespace``.

        This provides backward compatibility with code that accesses
        ``args.lr``, ``args.aggr``, etc.
        """
        ns = argparse.Namespace()

        # --- Experiment ---
        ns.trials = self.experiment.trials
        ns.gpu_id = self.experiment.gpu_id
        ns.save_loc = self.experiment.save_loc

        # --- Federation ---
        ns.global_epoch = self.federation.global_epoch
        ns.localIter = self.federation.local_iter
        ns.num_client = self.federation.num_clients
        ns.cl_part = self.federation.participation_ratio
        ns.traitor = self.federation.traitor_ratio
        ns.attack = self.federation.attack
        ns.aggr = self.federation.aggregator
        ns.hybrid_aggregator_list = self.federation.hybrid_aggregator_list
        ns.embd_momentum = self.federation.embedded_momentum
        ns.early_stop = self.federation.early_stop
        ns.MITM = self.federation.mitm
        ns.bucketing = self.federation.bucketing
        ns.bucket_type = self.federation.bucket_type
        ns.bucket_size = self.federation.bucket_size

        # --- Defense ---
        ns.tau = self.defense.tau
        ns.n_iter = self.defense.n_iter
        ns.buck_rand = self.defense.buck_rand
        ns.buck_len = self.defense.buck_len
        ns.buck_len_ecc = self.defense.buck_len_ecc
        ns.buck_avg = self.defense.buck_avg
        ns.multi_clip = self.defense.multi_clip
        ns.bucket_op = self.defense.bucket_op
        ns.ref_fixed = self.defense.ref_fixed
        ns.shuffle_bucket_order = self.defense.shuffle_bucket_order
        ns.combine_bucket = self.defense.combine_bucket
        ns.T = self.defense.T
        ns.nu = self.defense.nu
        ns.gas_p = self.defense.gas_p
        ns.fg_use_memory = self.defense.fg_use_memory
        ns.fg_memory_size = self.defense.fg_memory_size
        ns.fg_epsilon = self.defense.fg_epsilon
        ns.cheby_k_sigma = self.defense.cheby_k_sigma
        ns.foundation_num_synthetic = self.defense.foundation_num_synthetic
        ns.lalambda_n = self.defense.lalambda_n
        ns.lalambda_s = self.defense.lalambda_s
        ns.lasa_sparsity_ratio = self.defense.lasa_sparsity_ratio
        ns.fedseca_sparsity_gamma = self.defense.fedseca_sparsity_gamma
        ns.fedredefense_n_components = self.defense.fedredefense_n_components
        ns.fedredefense_threshold = self.defense.fedredefense_threshold
        ns.skymask_lr = self.defense.skymask_lr
        ns.skymask_epochs = self.defense.skymask_epochs
        ns.fldetector_warmup = self.defense.fldetector_warmup
        ns.fldetector_lbfgs_history = self.defense.fldetector_lbfgs_history
        ns.flame_epsilon = self.defense.flame_epsilon
        ns.flame_delta = self.defense.flame_delta
        ns.flame_add_noise = self.defense.flame_add_noise
        ns.dnc_sub_dim = self.defense.dnc_sub_dim
        ns.dnc_num_iters = self.defense.dnc_num_iters
        ns.dnc_filter_frac = self.defense.dnc_filter_frac
        ns.num_clustering = self.defense.num_clustering
        ns.bucket_shift = self.defense.bucket_shift
        ns.shift_amount = self.defense.shift_amount
        ns.buck_len_l2 = self.defense.buck_len_l2
        ns.apply_TM = self.defense.apply_TM
        ns.seq_update = self.defense.seq_update

        # --- Attack ---
        ns.z_max = self.attack.z_max
        ns.alie_z_max = self.attack.alie_z_max
        ns.nestrov_attack = self.attack.nestrov_attack
        ns.epsilon = self.attack.epsilon
        ns.pert_vec = self.attack.pert_vec
        ns.delta_coeff = self.attack.delta_coeff
        ns.lasa_attack_k1 = self.attack.lasa_attack_k1
        ns.lasa_attack_k2 = self.attack.lasa_attack_k2
        ns.pi = self.attack.pi
        ns.angle = self.attack.angle
        ns.lamb = self.attack.lamb

        # --- Optimizer ---
        ns.opt = self.optimizer.name
        ns.max_grad_norm = self.optimizer.max_grad_norm
        ns.lr = self.optimizer.lr
        ns.lr_decay = self.optimizer.lr_decay_epochs
        ns.wd = self.optimizer.weight_decay
        ns.Lmomentum = self.optimizer.momentum
        ns.betas = self.optimizer.betas
        ns.worker_momentum = self.optimizer.worker_momentum
        ns.nesterov = self.optimizer.nesterov

        # --- Model / Dataset ---
        ns.dataset_name = self.model.dataset_name
        ns.dataset_dist = self.model.dataset_dist
        ns.numb_cls_usr = self.model.num_classes_per_user
        ns.alpha = self.model.dirichlet_alpha
        ns.bs = self.model.batch_size
        ns.num_workers = self.model.num_workers
        ns.nn_name = self.model.nn_name
        ns.weight_init = self.model.weight_init
        ns.norm_type = self.model.norm_type
        ns.num_groups = self.model.num_groups

        # --- Pruning ---
        ns.load_mask = self.pruning.load_mask
        ns.prune_dataset_split = self.pruning.prune_dataset_split
        ns.omniscient_pruning = self.pruning.omniscient_pruning
        ns.sparse_cfg = self.pruning.sparse_cfg
        ns.pruning_factor = self.pruning.pruning_factor
        ns.sparse_scale = self.pruning.sparse_scale
        ns.sparse_sign = self.pruning.sparse_sign
        ns.sparse_th = self.pruning.sparse_th
        ns.prune_method = self.pruning.prune_method
        ns.prune_bias = self.pruning.prune_bias
        ns.prune_bn = self.pruning.prune_bn
        ns.keep_orig_weights = self.pruning.keep_orig_weights
        ns.first_layer_constraint = self.pruning.first_layer_constraint
        ns.last_layer_constraint = self.pruning.last_layer_constraint
        ns.min_threshold = self.pruning.min_threshold
        ns.inout_layers = self.pruning.inout_layers
        ns.num_steps = self.pruning.num_steps
        ns.prune_bs = self.pruning.prune_bs
        ns.mode = self.pruning.mode
        ns.num_batches = self.pruning.num_batches
        ns.force_w = self.pruning.force_w
        ns.force_g = self.pruning.force_g
        ns.force_v = self.pruning.force_v
        ns.init = self.pruning.init
        ns.mask_scope = self.pruning.mask_scope

        return ns


# ── CLI parser ───────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser matching the old ``parameters.py`` flags."""
    parser = argparse.ArgumentParser(
        description='FL-Byzantine-Library: Byzantine-resilient Federated Learning'
    )

    # --- Experiment ---
    g = parser.add_argument_group('Experiment')
    g.add_argument('--trials', type=int, default=1, help='Number of trials')
    g.add_argument('--gpu_id', type=int, default=0, help='GPU device id (-1 for CPU)')
    g.add_argument('--save_loc', type=str, default='', help='Save location for results')

    # --- Federation ---
    g = parser.add_argument_group('Federation')
    g.add_argument('--global_epoch', type=int, default=100, help='Communication rounds')
    g.add_argument('--localIter', type=int, default=1, help='Local epochs per round')
    g.add_argument('--num_client', type=int, default=25, help='Number of clients')
    g.add_argument('--cl_part', type=float, default=1., help='Participation ratio')
    g.add_argument('--traitor', type=float, default=0.2, help='Traitor ratio')
    g.add_argument('--attack', type=str, default='lasa', help='Attack strategy')
    g.add_argument('--aggr', type=str, default='lasa', help='Aggregation rule')
    g.add_argument('--hybrid_aggregator_list', type=str, default='cc+tm',
                   help='Aggregators for hybrid aggregation (+-separated)')
    g.add_argument('--embd_momentum', type=bool, default=False, help='FedADC embedded momentum')
    g.add_argument('--early_stop', type=bool, default=False, help='Early stopping')
    g.add_argument('--MITM', type=bool, default=True, help='Man-in-the-middle adversary')
    g.add_argument('--bucketing', type=bool, default=False, help='Enable bucketing')
    g.add_argument('--bucket_type', type=str, default='Random', help='Bucketing strategy')
    g.add_argument('--bucket_size', type=int, default=3, help='Bucket size')

    # --- Defense ---
    g = parser.add_argument_group('Defense')
    g.add_argument('--tau', type=float, default=1.0, help='CC clipping radius')
    g.add_argument('--n_iter', type=int, default=1, help='CC iterations')
    g.add_argument('--buck_rand', type=bool, default=False)
    g.add_argument('--buck_len', type=int, default=3, help='Bucket length for sequential CC')
    g.add_argument('--buck_len_ecc', type=int, default=3)
    g.add_argument('--buck_avg', type=bool, default=False)
    g.add_argument('--multi_clip', type=bool, default=False)
    g.add_argument('--bucket_op', type=str, default=None, help='Bucket remainder op')
    g.add_argument('--ref_fixed', type=bool, default=False)
    g.add_argument('--shuffle_bucket_order', type=bool, default=False)
    g.add_argument('--combine_bucket', type=bool, default=False)
    g.add_argument('--T', type=int, default=3, help='RFA inner iterations')
    g.add_argument('--nu', type=float, default=0.1, help='RFA norm budget')
    g.add_argument('--gas_p', type=int, default=1000, help='GAS chunks')
    g.add_argument('--fg_use_memory', type=bool, default=True)
    g.add_argument('--fg_memory_size', type=int, default=10)
    g.add_argument('--fg_epsilon', type=float, default=1e-5)
    g.add_argument('--cheby_k_sigma', type=float, default=1.0)
    g.add_argument('--foundation_num_synthetic', type=int, default=2)
    g.add_argument('--lalambda_n', type=float, default=1.0)
    g.add_argument('--lalambda_s', type=float, default=1.0)
    g.add_argument('--lasa_sparsity_ratio', type=float, default=0.7)
    g.add_argument('--fedseca_sparsity_gamma', type=float, default=0.9)
    g.add_argument('--fedredefense_n_components', type=int, default=3)
    g.add_argument('--fedredefense_threshold', type=float, default=0.6)
    g.add_argument('--skymask_lr', type=float, default=0.01)
    g.add_argument('--skymask_epochs', type=int, default=20)
    g.add_argument('--fldetector_warmup', type=int, default=10)
    g.add_argument('--fldetector_lbfgs_history', type=int, default=5)
    g.add_argument('--flame_epsilon', type=float, default=3000)
    g.add_argument('--flame_delta', type=float, default=0.01)
    g.add_argument('--flame_add_noise', type=bool, default=True)
    g.add_argument('--dnc_sub_dim', type=int, default=10000)
    g.add_argument('--dnc_num_iters', type=int, default=1)
    g.add_argument('--dnc_filter_frac', type=float, default=1.0)
    g.add_argument('--num_clustering', type=int, default=3)
    g.add_argument('--bucket_shift', type=str, default='sequential')
    g.add_argument('--shift_amount', type=int, default=1)
    g.add_argument('--buck_len_l2', type=int, default=3)
    g.add_argument('--apply_TM', type=bool, default=False)
    g.add_argument('--seq_update', type=bool, default=False)

    # --- Attack ---
    g = parser.add_argument_group('Attack')
    g.add_argument('--z_max', type=float, default=None, help='Attack scale')
    g.add_argument('--alie_z_max', type=float, default=None, help='ALIE attack scale')
    g.add_argument('--nestrov_attack', type=bool, default=False)
    g.add_argument('--epsilon', type=float, default=0.2, help='IPM scale')
    g.add_argument('--pert_vec', type=str, default='std', help='Perturbation vector type')
    g.add_argument('--delta_coeff', type=float, default=0.9)
    g.add_argument('--lasa_attack_k1', type=float, default=0.01)
    g.add_argument('--lasa_attack_k2', type=float, default=0.7)
    g.add_argument('--pi', type=float, default=1, help='ROP location')
    g.add_argument('--angle', type=float, default=270, help='ROP angle')
    g.add_argument('--lamb', type=float, default=0.9, help='ROP reference')

    # --- Optimizer ---
    g = parser.add_argument_group('Optimizer')
    g.add_argument('--opt', type=str, default='sgd', help='Optimizer name')
    g.add_argument('--max_grad_norm', type=float, default=-1, help='Grad clipping norm')
    g.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    g.add_argument('--lr_decay', type=float, default=[75], help='LR decay epochs')
    g.add_argument('--wd', type=float, default=0, help='Weight decay')
    g.add_argument('--Lmomentum', type=float, default=0.9, help='SGD momentum')
    g.add_argument('--betas', type=tuple, default=(0.9, 0.999), help='Adam betas')
    g.add_argument('--worker_momentum', type=bool, default=True)
    g.add_argument('--nesterov', type=bool, default=False)

    # --- Model / Dataset ---
    g = parser.add_argument_group('Model & Dataset')
    g.add_argument('--dataset_name', type=str, default='cifar10')
    g.add_argument('--dataset_dist', type=str, default='iid', help='Data distribution')
    g.add_argument('--numb_cls_usr', type=int, default=2, help='Classes per client')
    g.add_argument('--alpha', type=float, default=1., help='Dirichlet alpha')
    g.add_argument('--bs', type=int, default=32, help='Batch size')
    g.add_argument('--num_workers', type=int, default=0, help='DataLoader workers')
    g.add_argument('--nn_name', type=str, default='resnet20', help='Model name')
    g.add_argument('--weight_init', type=str, default='-', help='Weight init')
    g.add_argument('--norm_type', type=str, default='bn', help='Normalization')
    g.add_argument('--num_groups', type=int, default=32, help='GroupNorm groups')

    # --- Pruning ---
    g = parser.add_argument_group('Pruning')
    g.add_argument('--load_mask', type=str, default=None)
    g.add_argument('--prune_dataset_split', type=float, default=1.)
    g.add_argument('--omniscient_pruning', type=bool, default=True)
    g.add_argument('--sparse_cfg', type=int, default=50)
    g.add_argument('--pruning_factor', type=float, default=0.005, dest='pruning_factor')
    g.add_argument('--sparse_scale', type=float, default=1.5)
    g.add_argument('--sparse_sign', type=str, default='inv_std')
    g.add_argument('--sparse_th', type=str, default=None)
    g.add_argument('--prune_method', type=str, default='force', dest='prune_method')
    g.add_argument('--prune_bias', type=bool, default=False)
    g.add_argument('--prune_bn', type=bool, default=False)
    g.add_argument('--keep_orig_weights', type=bool, default=True)
    g.add_argument('--first_layer_constraint', type=int, default=-1)
    g.add_argument('--last_layer_constraint', type=int, default=-1)
    g.add_argument('--min_threshold', type=float, default=-1)
    g.add_argument('--inout_layers', type=bool, default=False)
    g.add_argument('--num_steps', type=int, default=100)
    g.add_argument('--prune_bs', type=int, default=32)
    g.add_argument('--mode', type=str, default='exp')
    g.add_argument('--num_batches', type=int, default=3)
    g.add_argument('--force_w', type=float, default=1.)
    g.add_argument('--force_g', type=float, default=1.)
    g.add_argument('--force_v', type=float, default=1.)
    g.add_argument('--init', type=str, default='-')
    g.add_argument('--mask_scope', type=str, default='local')

    return parser


def _namespace_to_config(ns: argparse.Namespace) -> FLConfig:
    """Map a flat ``argparse.Namespace`` into a structured ``FLConfig``."""
    return FLConfig(
        experiment=ExperimentConfig(
            trials=ns.trials,
            gpu_id=ns.gpu_id,
            save_loc=ns.save_loc,
        ),
        federation=FederationConfig(
            global_epoch=ns.global_epoch,
            local_iter=ns.localIter,
            num_clients=ns.num_client,
            participation_ratio=ns.cl_part,
            traitor_ratio=ns.traitor,
            attack=ns.attack,
            aggregator=ns.aggr,
            hybrid_aggregator_list=ns.hybrid_aggregator_list,
            embedded_momentum=ns.embd_momentum,
            early_stop=ns.early_stop,
            mitm=ns.MITM,
            bucketing=ns.bucketing,
            bucket_type=ns.bucket_type,
            bucket_size=ns.bucket_size,
        ),
        optimizer=OptimizerConfig(
            name=ns.opt,
            lr=ns.lr,
            lr_decay_epochs=ns.lr_decay if isinstance(ns.lr_decay, list) else [int(ns.lr_decay)],
            weight_decay=ns.wd,
            momentum=ns.Lmomentum,
            betas=ns.betas,
            max_grad_norm=ns.max_grad_norm,
            worker_momentum=ns.worker_momentum,
            nesterov=ns.nesterov,
        ),
        model=ModelConfig(
            dataset_name=ns.dataset_name,
            dataset_dist=ns.dataset_dist,
            num_classes_per_user=ns.numb_cls_usr,
            dirichlet_alpha=ns.alpha,
            batch_size=ns.bs,
            num_workers=ns.num_workers,
            nn_name=ns.nn_name,
            weight_init=ns.weight_init,
            norm_type=ns.norm_type,
            num_groups=ns.num_groups,
        ),
        defense=DefenseConfig(
            tau=ns.tau,
            n_iter=ns.n_iter,
            buck_rand=ns.buck_rand,
            buck_len=ns.buck_len,
            buck_len_ecc=ns.buck_len_ecc,
            buck_avg=ns.buck_avg,
            multi_clip=ns.multi_clip,
            bucket_op=ns.bucket_op,
            ref_fixed=ns.ref_fixed,
            shuffle_bucket_order=ns.shuffle_bucket_order,
            combine_bucket=ns.combine_bucket,
            T=ns.T,
            nu=ns.nu,
            gas_p=ns.gas_p,
            fg_use_memory=ns.fg_use_memory,
            fg_memory_size=ns.fg_memory_size,
            fg_epsilon=ns.fg_epsilon,
            cheby_k_sigma=ns.cheby_k_sigma,
            foundation_num_synthetic=ns.foundation_num_synthetic,
            lalambda_n=ns.lalambda_n,
            lalambda_s=ns.lalambda_s,
            lasa_sparsity_ratio=ns.lasa_sparsity_ratio,
            fedseca_sparsity_gamma=ns.fedseca_sparsity_gamma,
            fedredefense_n_components=ns.fedredefense_n_components,
            fedredefense_threshold=ns.fedredefense_threshold,
            skymask_lr=ns.skymask_lr,
            skymask_epochs=ns.skymask_epochs,
            fldetector_warmup=ns.fldetector_warmup,
            fldetector_lbfgs_history=ns.fldetector_lbfgs_history,
            flame_epsilon=ns.flame_epsilon,
            flame_delta=ns.flame_delta,
            flame_add_noise=ns.flame_add_noise,
            dnc_sub_dim=ns.dnc_sub_dim,
            dnc_num_iters=ns.dnc_num_iters,
            dnc_filter_frac=ns.dnc_filter_frac,
            num_clustering=ns.num_clustering,
            bucket_shift=ns.bucket_shift,
            shift_amount=ns.shift_amount,
            buck_len_l2=ns.buck_len_l2,
            apply_TM=ns.apply_TM,
            seq_update=ns.seq_update,
        ),
        attack=AttackConfig(
            z_max=ns.z_max,
            alie_z_max=ns.alie_z_max,
            nestrov_attack=ns.nestrov_attack,
            epsilon=ns.epsilon,
            pert_vec=ns.pert_vec,
            delta_coeff=ns.delta_coeff,
            lasa_attack_k1=ns.lasa_attack_k1,
            lasa_attack_k2=ns.lasa_attack_k2,
            pi=ns.pi,
            angle=ns.angle,
            lamb=ns.lamb,
        ),
        pruning=PruningConfig(
            load_mask=ns.load_mask,
            prune_dataset_split=ns.prune_dataset_split,
            omniscient_pruning=ns.omniscient_pruning,
            sparse_cfg=ns.sparse_cfg,
            pruning_factor=ns.pruning_factor,
            sparse_scale=ns.sparse_scale,
            sparse_sign=ns.sparse_sign,
            sparse_th=ns.sparse_th,
            prune_method=ns.prune_method,
            prune_bias=ns.prune_bias,
            prune_bn=ns.prune_bn,
            keep_orig_weights=ns.keep_orig_weights,
            first_layer_constraint=ns.first_layer_constraint,
            last_layer_constraint=ns.last_layer_constraint,
            min_threshold=ns.min_threshold,
            inout_layers=ns.inout_layers,
            num_steps=ns.num_steps,
            prune_bs=ns.prune_bs,
            mode=ns.mode,
            num_batches=ns.num_batches,
            force_w=ns.force_w,
            force_g=ns.force_g,
            force_v=ns.force_v,
            init=ns.init,
            mask_scope=ns.mask_scope,
        ),
    )


# ── Public API ───────────────────────────────────────────────────────────────

def parse_args() -> FLConfig:
    """Parse command-line arguments into a structured ``FLConfig``."""
    parser = _build_parser()
    ns = parser.parse_args()
    return _namespace_to_config(ns)


def args_parser() -> argparse.Namespace:
    """**Legacy compatibility entry point.**

    Returns a flat ``argparse.Namespace`` identical to the old
    ``parameters.args_parser()`` so existing code works without changes.
    """
    parser = _build_parser()
    return parser.parse_args()
