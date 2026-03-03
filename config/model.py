"""Neural network and dataset configuration."""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Dataset, model architecture, and data-loading settings."""

    # --- Dataset ---
    dataset_name: str = 'cifar10'
    """Dataset identifier: cifar10, cifar100, mnist, fmnist, emnist-d,
    emnist-l, emnist-b, svhn, tiny_imagenet, imagenette."""

    dataset_dist: str = 'iid'
    """Data distribution across clients: 'iid', 'sort_part', or 'dirichlet'."""

    num_classes_per_user: int = 2
    """Number of label classes per client when using sort_part distribution."""

    dirichlet_alpha: float = 1.0
    """Dirichlet concentration parameter. Lower = more skewed."""

    batch_size: int = 32
    """Training batch size per client."""

    num_workers: int = 0
    """Number of DataLoader worker processes for all data loaders."""

    # --- Model architecture ---
    nn_name: str = 'resnet20'
    """Model name: simplecnn, simplecifar, resnet{8,9,18,20},
    vgg{11,13,16,19}, mobilenet, efficientnet, mnistnet, mlp_big, mlp_small,
    mobilevit_{xxs,xs,s}."""

    weight_init: str = '-'
    """Weight initialization: 'kn' (Kaiming normal) or '-' (default)."""

    norm_type: str = 'bn'
    """Normalization layer: 'bn' (BatchNorm), 'gn' (GroupNorm), '-' (None)."""

    num_groups: int = 32
    """Number of groups for GroupNorm. Use 1 for LayerNorm."""
