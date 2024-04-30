# Byzantine attacks and defenses in federated learning Library

This library contains the implementation of the Byzantine attacks and defenses in federated learning (FL).

## Aggregators:
- Aggregators can be extended by adding the aggregator in the `aggregators` folder.


- [x] **Bulyan** - The Hidden Vulnerability of Distributed Learning in Byzantium [[ICML 2018]](https://proceedings.mlr.press/v80/mhamdi18a.html)
- [x] **Centered Clipping** - [[ICML 2021]](http://proceedings.mlr.press/v139/karimireddy21a.html)
- [x] **Centered Median** - 
- [x] **Krum** - 
- [x] **Trimmed Mean** - 
- [x] **SignSGD** - 
- [x] **RFA** - 
- [x] **Sequantial Centered Clipping** -  
- [x] **FL-Trust** - 
- [x] **GAS (Krum and Bulyan)** - 
- [x] **FedAVG** -


## Byzantine Attacks:
- Attacks can be extended by adding the attack in the `attacks` folder.


- [x] **Label-Flip** -
- [x] **Bit-Flip** - 
- [x] **Gaussian noise** - 
- [x] **Untargeted C&W** - 
- [x] **Little is enough (ALIE)** - 
- [x] **Inner product Manipulation (IPM)** - 
- [x] **Relocated orthogonal perturbation (ROP)** - 
- [x] **Min-sum** - 
- [x] **Min-max** - 
- [x] **Sparse** - 
- [x] **Sparse-Optimized** - 


## Datasets:
- [x] **MNIST**
- [x] **CIFAR-10**
- [x] **CIFAR-100**
- [x] **Fashion-MNIST**
- [x] **EMNIST**
- [x] **SVHN**
- [x] **Tiny-ImageNet**

Datasets can be extended by adding the dataset in the `datasets` folder. Any labeled vision classification dataset in https://pytorch.org/vision/main/datasets.html can be used.


### Available data distributions:
- [x] **IID**
- [x] **Non-IID**: 
    - [x] **Dirichlet** lower the alpha, more non-IID the data becomes. value "1" generally realistic for the real FL scenarios.
    - [x] **Sort-and-Partition** Distributes only a few selected classes to each client.

## Models:
- Models can be extended by adding the model in the `models` folder and by modifying the 'nn_classes' accordingly.
- Different Norms and initialization functions are available in 'nn_classes.


### Available models:
- [x] **MLP** Different sizes of MLP models are available for grayscale images.
- [x] **CNN (various sizes)** Different CNN models are available for RGB and grayscale images respectively
- [x] **ResNet** RGB datasets only. Various depts and sizes are available (8-20-9-18).
- [x] **VGG** RGB datasets only. Various depts and sizes are available.
- [x] **MobileNet** RGB datasets only.

### Future models:
- [x] **Visual Transformers** (ViT , DeiT, Swin, Twin, etc.) 


## Installation

1. Install Python 3.8. For convenience, execute the following command.

```
pip install -r requirements.txt
```

## Citation

If you find this repo useful, please cite our papers.

```
@ARTICLE{10366296,
  author={Özfatura, Kerem and Özfatura, Emre and Küpçü, Alptekin and Gunduz, Deniz},
  journal={IEEE Transactions on Information Forensics and Security}, 
  title={Byzantines Can Also Learn From History: Fall of Centered Clipping in Federated Learning}, 
  year={2024},
  volume={19},
  number={},
  pages={2010-2022},
  keywords={Task analysis;Robustness;Federated learning;Security;Training;Aggregates;Taxonomy;Federated learning;adversarial machine learning;deep learning},
  doi={10.1109/TIFS.2023.3345171}}
```

```
@misc{ozfatura2024aggressive,
      title={Aggressive or Imperceptible, or Both: Network Pruning Assisted Hybrid Byzantines in Federated Learning}, 
      author={Emre Ozfatura and Kerem Ozfatura and Alptekin Kupcu and Deniz Gunduz},
      year={2024},
      eprint={2404.06230},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## Contact
If you have any questions or suggestions, feel free to contact:

- Kerem Özfatura (aozfatura22@ku.edu.tr)
- Emre Özfatura (m.ozfatura@imperial.ac.uk)
