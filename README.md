# FL-Byzantine-Library

A modular Python library for **Byzantine-resilient Federated Learning** research.
It provides a plug-and-play framework for evaluating robust aggregation rules against a wide range of Byzantine attacks under realistic federated settings.

---

## Table of Contents

- [Introduction](#introduction)
- [Library Architecture](#library-architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Aggregators (Defenses)](#aggregators-defenses)
- [Byzantine Attacks](#byzantine-attacks)
- [Datasets](#datasets)
- [Models](#models)
- [Configuration System](#configuration-system)
- [Extending the Library](#extending-the-library)
- [Citation](#citation)
- [Contact](#contact)

---

## Introduction

Federated Learning (FL) enables multiple participants to collaboratively train a model without sharing raw data.
However, FL is inherently vulnerable to **Byzantine attacks**, where malicious participants send corrupted model updates to degrade global model performance.

**FL-Byzantine-Library** provides:

- **22+ robust aggregation rules** — from classic methods (Krum, Trimmed Mean) to state-of-the-art defenses (LASA, FedSECA, SkyMask, FLAME).
- **16+ Byzantine attack strategies** — including model poisoning (Fang, LMP), data poisoning (Label-Flip), and optimization-based attacks (Min-Max, ALIE).
- **Structured configuration** — typed dataclass-based configs for reproducible experiments.
- **Modular design** — add new aggregators, attacks, models, or datasets with minimal boilerplate.
- **Network pruning** — integrated sparse attack and defense mechanisms with multiple pruning algorithms.

---

## Library Architecture

```
FL-Byzantine-Library/
├── main.py                 # Entry point — training loop orchestration
├── fl.py                   # FL coordinator (training, aggregation, evaluation)
├── client.py               # Client class (local training, optimizer logic)
├── mapper.py               # Factory — maps config to components
├── model_registry.py       # Model factory (ResNet, VGG, MobileNet, etc.)
├── data_loader.py          # Dataset loading and distribution strategies
├── logger.py               # Result saving and plotting
├── utils.py                # Shared utilities (flatten, evaluate, etc.)
│
├── config/                 # 📦 Typed configuration system
│   ├── __init__.py
│   ├── base.py             # ExperimentConfig (GPU, trials, save path)
│   ├── federation.py       # FederationConfig (rounds, clients, attack/aggr)
│   ├── optimizer.py        # OptimizerConfig (SGD/Adam/AdamW settings)
│   ├── model.py            # ModelConfig (dataset, architecture, num_workers)
│   ├── defense.py          # DefenseConfig (aggregator hyperparameters)
│   ├── attack.py           # AttackConfig (attack hyperparameters)
│   ├── pruning.py          # PruningConfig (pruning algorithm settings)
│   └── parser.py           # FLConfig composer + CLI parser
│
├── aggregators/            # 🛡️ Robust aggregation rules
│   ├── base.py             # _BaseAggregator abstract class
│   ├── aggr_mapper.py      # Name → class registry
│   ├── krum.py, bulyan.py, trimmed_mean.py, cm.py, ...
│   └── ...
│
├── attacks/                # ⚔️ Byzantine attack strategies
│   ├── base.py             # _BaseByzantine abstract class
│   ├── attack_mapper.py    # Name → class registry
│   ├── alie.py, fang.py, minmax.py, ...
│   └── ...
│
├── models/                 # 🧠 Neural network architectures
│   ├── CNN.py, ResNet.py, VGG.py, MLP.py, ...
│   └── mobilevit.py
│
├── datasets/               # 📊 Dataset loaders
│   ├── RGB.py              # CIFAR-10/100, SVHN, Tiny-ImageNet, etc.
│   └── BW.py               # MNIST, Fashion-MNIST, EMNIST
│
├── pruners/                # ✂️ Network pruning algorithms
│   ├── prune_basic.py, synflow.py, erk.py, ...
│   └── prune_mapper.py
│
├── requirements.txt
└── setup.py
```

### How It Works

```
┌─────────────┐     CLI / Python API      ┌───────────────┐
│   main.py   │ ◀────── args ──────────── │ config/parser  │
└──────┬──────┘                            └───────────────┘
       │
       ▼
┌─────────────┐   creates clients,    ┌──────────────────┐
│  mapper.py  │ ─── aggregator, ────▶ │ FL coordinator   │
└─────────────┘   attack objects       │   (fl.py)        │
                                       └────────┬─────────┘
                                                │
                       ┌────────────────────────┼────────────────────────┐
                       ▼                        ▼                        ▼
               ┌──────────────┐        ┌──────────────┐        ┌──────────────┐
               │ Benign Client│        │ Benign Client│   ...  │ Malicious    │
               │  (client.py) │        │  (client.py) │        │  (attacks/)  │
               └──────────────┘        └──────────────┘        └──────────────┘
                       │                        │                        │
                       └────────────────────────┼────────────────────────┘
                                                │  gradients
                                                ▼
                                       ┌──────────────────┐
                                       │  Aggregator      │
                                       │  (aggregators/)  │
                                       └──────────────────┘
                                                │
                                                ▼
                                         Global Model Update
```

---

## Installation

### From source (development)

```bash
git clone https://github.com/CRYPTO-KU/FL-Byzantine-Library.git
cd FL-Byzantine-Library
pip install -r requirements.txt
```

### As a package

```bash
pip install .
```

---

## Quick Start

### 1. Command-Line Interface

```bash
# Show all available options
python main.py --help

# Basic run: 10 clients, CIFAR-10, Krum defense vs. ALIE attack
python main.py --num_client 10 --dataset_name cifar10 --aggr krum --attack alie

# Non-IID Dirichlet distribution with ResNet-20
python main.py --num_client 25 --dataset_name cifar10 --nn_name resnet20 \
    --dataset_dist dirichlet --alpha 0.5 --aggr tm --attack minmax \
    --global_epoch 200 --traitor 0.2

# Cross-device setting (partial participation)
python main.py --num_client 100 --cl_part 0.1 --aggr cc --attack rop

# With network pruning and sparse attacks
python main.py --aggr lasa --attack sparse --pruning_factor 0.01 \
    --prune_method force --num_client 25

# Multi-worker DataLoaders for faster training
python main.py --num_workers 4 --aggr avg --attack label_flip
```

### 2. After installing as a package

```bash
fl-byzantine --num_client 10 --aggr krum --attack alie --trials 3
```

### 3. Python API

```python
from config import FLConfig, ExperimentConfig, FederationConfig, ModelConfig

# Build a structured config programmatically
config = FLConfig(
    experiment=ExperimentConfig(trials=1, gpu_id=0),
    federation=FederationConfig(
        global_epoch=100,
        num_clients=25,
        traitor_ratio=0.2,
        aggregator='krum',
        attack='alie',
    ),
    model=ModelConfig(
        dataset_name='cifar10',
        nn_name='resnet20',
        batch_size=32,
        num_workers=4,
    ),
)

# Convert to flat namespace for backward-compatible code
args = config.to_flat_namespace()

# Use with existing library components
from mapper import Mapper
mapper = Mapper(args)
fl_instance = mapper.initialize_FL()
```

---

## Aggregators (Defenses)

All aggregators inherit from `aggregators.base._BaseAggregator`. Use the `--aggr` flag to select one.

| Key | Aggregator | Paper | Venue |
|-----|-----------|-------|-------|
| `avg` | **FedAVG** | Communication-Efficient Learning of Deep Networks from Decentralized Data | [AISTATS 2017](http://proceedings.mlr.press/v54/mcmahan17a.html) |
| `krum` | **Krum** | Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent | [NeurIPS 2017](https://proceedings.neurips.cc/paper/2017/file/f4b9ec30ad9f68f89b29639786cb62ef-Paper.pdf) |
| `bulyan` | **Bulyan** | The Hidden Vulnerability of Distributed Learning in Byzantium | [ICML 2018](https://proceedings.mlr.press/v80/mhamdi18a.html) |
| `tm` | **Trimmed Mean** | Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates | [ICML 2018](http://proceedings.mlr.press/v80/yin18a.html) |
| `cm` | **Centered Median** | Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates | [ICML 2018](http://proceedings.mlr.press/v80/yin18a.html) |
| `cc` | **Centered Clipping** | Learning from History for Byzantine Robust Optimization | [ICML 2021](http://proceedings.mlr.press/v139/karimireddy21a.html) |
| `scc` | **Sequential CC** | Byzantines Can Also Learn From History: Fall of Centered Clipping in FL | [IEEE TIFS 2024](https://ieeexplore.ieee.org/document/9636827) |
| `sign` | **SignSGD** | signSGD with Majority Vote is Communication Efficient and Fault Tolerant | [ICLR 2019](https://openreview.net/pdf?id=BJxhijAcY7) |
| `rfa` | **RFA** | Robust Aggregation for Federated Learning | [IEEE TSP 2022](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9721118) |
| `fl_trust` | **FL-Trust** | FLTrust: Byzantine-robust Federated Learning via Trust Bootstrapping | [NDSS 2021](https://www.ndss-symposium.org/wp-content/uploads/ndss2021_6C-2_24434_paper.pdf) |
| `gas` | **GAS** | Byzantine-robust Learning on Heterogeneous Data via Gradient Splitting | [ICML 2023](https://proceedings.mlr.press/v202/liu23d.html) |
| `foolsgold` | **FoolsGold** | Mitigating Sybils in Federated Learning Poisoning | [arXiv:1808.04866](https://arxiv.org/abs/1808.04866) |
| `dnc` | **DnC** | Manipulating the Byzantine: Optimizing Model Poisoning Attacks and Defenses | [NDSS 2021](https://par.nsf.gov/servlets/purl/10286354) |
| `flame` | **FLAME** | FLAME: Taming Backdoors in Federated Learning | [USENIX Security 2022](https://www.usenix.org/conference/usenixsecurity22/presentation/nguyen) |
| `fldetector` | **FLDetector** | Defending Federated Learning Against Model Poisoning via Detecting Malicious Clients | [KDD 2022](https://github.com/zaixizhang/FLDetector) |
| `skymask` | **SkyMask** | Attack-agnostic Robust Federated Learning with Fine-grained Learnable Masks | [ECCV 2024](https://github.com/KoalaYan/SkyMask) |
| `fl_defender` | **FL-Defender** | Combating Targeted Model Poisoning Attacks in Federated Learning | [GitHub](https://github.com/najeebjebreel/FL-Defender) |
| `fedredefense` | **FedREDefense** | Defending against Model Poisoning Attacks via Model Update Reconstruction Error | [ICML 2024](https://github.com/ShuangtongLi/FedREDefense) |
| `foundation` | **FoundationFL** | Do We Really Need to Design New Byzantine-robust Aggregation Rules? | [NDSS 2025](https://www.ndss-symposium.org/ndss-paper/do-we-really-need-to-design-new-byzantine-robust-aggregation-rules/) |
| `signguard` | **SignGuard** | Byzantine-robust Aggregation using Norm Filtering and Sign Clustering | [GitHub](https://github.com/JianXu95/SignGuard) |
| `lasa` | **LASA** | Achieving Byzantine-Resilient FL via Layer-Adaptive Sparsified Model Aggregation | [WACV 2025](https://github.com/CRYPTO-KU/LASA) |
| `fedseca` | **FedSECA** | FedSECA: Sign Election and Coordinate-wise Aggregation for Byzantine Tolerant FL | [CVPR 2025](https://github.com/CRYPTO-KU/FedSECA) |

Additional variants: `tm_cheby`, `tm_capped`, `tm_abs`, `tm_history`, `tm_perfect`, `med_krum`, `scc_krum`, `hybrid`, `adaptive_hybrid`, `cc_cluster`, `ccs_rand`, `ccs_ecc`, `foundation_tm`, `foundation_med`.

---

## Byzantine Attacks

All attacks inherit from `attacks.base._BaseByzantine`. Use the `--attack` flag to select one.

| Key | Attack | Paper | Venue |
|-----|--------|-------|-------|
| `label_flip` | **Label-Flip** | Poisoning Attacks against Support Vector Machines | [ICML 2012](https://icml.cc/2012/papers/880.pdf) |
| `bit_flip` | **Bit-Flip (PGD)** | — | — |
| `alie` | **ALIE** | A Little Is Enough: Circumventing Defenses For Distributed Learning | [NeurIPS 2019](https://proceedings.neurips.cc/paper/2019/file/ec1c59141046cd1866bbbcdfb6ae31d4-Paper.pdf) |
| `ipm` | **IPM** | Fall of Empires: Breaking Byzantine-tolerant SGD by Inner Product Manipulation | [UAI 2019](http://auai.org/uai2019/proceedings/papers/83.pdf) |
| `reloc` | **ROP** | Byzantines Can Also Learn From History: Fall of Centered Clipping in FL | [IEEE TIFS 2024](https://ieeexplore.ieee.org/document/9636827) |
| `minmax` | **Min-Max** | Manipulating the Byzantine: Optimizing Model Poisoning Attacks and Defenses | [NDSS 2021](https://par.nsf.gov/servlets/purl/10286354) |
| `minsum` | **Min-Sum** | Manipulating the Byzantine: Optimizing Model Poisoning Attacks and Defenses | [NDSS 2021](https://par.nsf.gov/servlets/purl/10286354) |
| `fang` | **Fang** | Local Model Poisoning Attacks to Byzantine-Robust Federated Learning | [USENIX Security 2020](https://www.usenix.org/conference/usenixsecurity20/presentation/fang) |
| `cw` | **C&W** | Towards Evaluating the Robustness of Neural Networks | [IEEE S&P 2017](https://ieeexplore.ieee.org/document/7958570) |
| `krum_attack` | **LMP-Krum** | Local Model Poisoning Attacks to Byzantine-Robust FL | [arXiv:1911.11815](https://arxiv.org/abs/1911.11815) |
| `trimmed_mean_attack` | **LMP-TM** | Local Model Poisoning Attacks to Byzantine-Robust FL | [arXiv:1911.11815](https://arxiv.org/abs/1911.11815) |
| `mimic` | **Mimic** | Byzantine-Robust Learning on Heterogeneous Datasets via Bucketing | [arXiv:2006.09365](https://arxiv.org/abs/2006.09365) |
| `sparse` | **Sparse** | Aggressive or Imperceptible, or Both: Network Pruning Assisted Hybrid Byzantines in Federated Learning | [arXiv:2404.06230](https://arxiv.org/abs/2404.06230) |
| `sparse_opt` | **Sparse-Optimized** | Aggressive or Imperceptible, or Both: Network Pruning Assisted Hybrid Byzantines in Federated Learning | [arXiv:2404.06230](https://arxiv.org/abs/2404.06230) |
| `lasa` | **LASA Attack** | Aggressive or Imperceptible, or Both: Network Pruning Assisted Hybrid Byzantines in Federated Learning| [WACV 2025](https://github.com/CRYPTO-KU/LASA) |

Additional variants: `edge_case`, `local_minmax`, `adaptive_krum`, `local_trimmed_mean`, `stealthy`, `mimic_variant`, `adaptive_mimic`.

---

## Datasets

| Dataset | Key | Type |
|---------|-----|------|
| CIFAR-10 | `cifar10` | RGB 32×32 |
| CIFAR-100 | `cifar100` | RGB 32×32 |
| SVHN | `svhn` | RGB 32×32 |
| Tiny-ImageNet | `tiny_imagenet` | RGB 64×64 |
| Imagenette | `imagenette` | RGB 224×224 |
| MNIST | `mnist` | Grayscale 28×28 |
| Fashion-MNIST | `fmnist` | Grayscale 28×28 |
| EMNIST (digits/letters/balanced) | `emnist-d` / `emnist-l` / `emnist-b` | Grayscale 28×28 |

Datasets can be extended by adding loaders in the `datasets/` folder. Any labeled vision classification dataset from [torchvision](https://pytorch.org/vision/main/datasets.html) can be integrated.

### Data Distributions

| Distribution | Key | Description |
|-------------|-----|-------------|
| IID | `iid` | Uniform random split |
| Dirichlet | `dirichlet` | Heterogeneous split controlled by `--alpha` (lower = more skewed) |
| Sort-and-Partition | `sort_part` | Each client gets only `--numb_cls_usr` label classes |

---

## Models

Models can be extended by adding them in `models/` and registering in `model_registry.py`.

| Model | Key | Input |
|-------|-----|-------|
| MLP (small/big) | `mlp_small`, `mlp_big` | Grayscale |
| Simple CNN | `simplecnn`, `simplecifar` | RGB / Grayscale |
| ResNet (8/9/18/20) | `resnet8`, `resnet9`, `resnet18`, `resnet20` | RGB |
| VGG (11/13/16/19) | `vgg11`, `vgg13`, `vgg16`, `vgg19` | RGB |
| MobileNet | `mobilenet` | RGB |
| EfficientNet-B0 | `efficientnet` | RGB |
| MobileViT (xxs/xs/s) | `mobilevit_xxs`, `mobilevit_xs`, `mobilevit_s` | RGB |
| MNIST-Net | `mnistnet` | Grayscale |

---

## Configuration System

The library uses a **dataclass-based configuration system** organized into logical groups:

| Config Class | Responsibility | Key Parameters |
|-------------|---------------|----------------|
| `ExperimentConfig` | Hardware & I/O | `trials`, `gpu_id`, `save_loc` |
| `FederationConfig` | FL protocol | `global_epoch`, `num_clients`, `traitor_ratio`, `aggregator`, `attack` |
| `OptimizerConfig` | Local optimizer | `lr`, `momentum`, `weight_decay`, `betas` |
| `ModelConfig` | Dataset & architecture | `dataset_name`, `nn_name`, `batch_size`, `num_workers` |
| `DefenseConfig` | Aggregator hyperparams | `tau`, `flame_epsilon`, `lasa_sparsity_ratio`, etc. |
| `AttackConfig` | Attack hyperparams | `z_max`, `epsilon`, `pert_vec`, etc. |
| `PruningConfig` | Pruning settings | `pruning_factor`, `prune_method`, `sparse_scale`, etc. |

All configs compose into a single `FLConfig`:

```python
from config import FLConfig, FederationConfig, DefenseConfig

config = FLConfig(
    federation=FederationConfig(aggregator='flame', attack='alie'),
    defense=DefenseConfig(flame_epsilon=5000, flame_delta=0.01),
)

# Use as flat namespace (backward compatible)
args = config.to_flat_namespace()
```

CLI flags map directly to the old argument names for full backward compatibility.

---

## Extending the Library

### Adding a New Aggregator

1. Create `aggregators/my_aggregator.py`:

```python
from .base import _BaseAggregator

class MyAggregator(_BaseAggregator):
    def __init__(self, **kwargs):
        super().__init__()
        # init params

    def __call__(self, inputs):
        # inputs: List[torch.Tensor] — flattened gradient from each client
        # return: aggregated gradient tensor
        return torch.mean(torch.stack(inputs), dim=0)
```

2. Register in `aggregators/aggr_mapper.py`:

```python
from .my_aggregator import MyAggregator
aggr_mapper['my_aggr'] = MyAggregator
```

3. Add parameters to `set_aggr_params()` in the same file.

### Adding a New Attack

1. Create `attacks/my_attack.py`:

```python
from .base import _BaseByzantine

class MyAttack(_BaseByzantine):
    def __init__(self, n, m, z, eps, layer_inds, **kwargs):
        super().__init__(**kwargs)

    def omniscient_callback(self, benign_gradients):
        # Craft adversarial gradient using benign_gradients
        mean = torch.mean(torch.stack(benign_gradients), dim=0)
        self.adv_momentum = -mean  # example: send negated mean
```

2. Register in `attacks/attack_mapper.py`:

```python
from .my_attack import MyAttack
attack_mapper['my_attack'] = MyAttack
```

---

## Citation

If you find this library useful, please cite our papers:

```bibtex
@ARTICLE{ROP,
  author={Ozfatura, Kerem and Ozfatura, Emre and Kupcu, Alptekin and Gunduz, Deniz},
  journal={IEEE Transactions on Information Forensics and Security},
  title={Byzantines Can Also Learn From History: Fall of Centered Clipping in Federated Learning},
  year={2024},
  volume={19},
  pages={2010-2022},
  doi={10.1109/TIFS.2023.3345171}
}
```

```bibtex
@misc{sparseATK,
  title={Aggressive or Imperceptible, or Both: Network Pruning Assisted Hybrid Byzantines in Federated Learning},
  author={Emre Ozfatura and Kerem Ozfatura and Alptekin Kupcu and Deniz Gunduz},
  year={2024},
  eprint={2404.06230},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}
```

---

## Contact

If you have any questions or suggestions, feel free to contact:

- **Kerem Özfatura** — [aozfatura22@ku.edu.tr](mailto:aozfatura22@ku.edu.tr)
