# Byzantine attacks and defenses in federated learning Library

This library contains the implementation of the Byzantine attacks and defenses in federated learning.

## Aggregators:
- Aggregators can be extended by adding the aggregator in the `aggregators` folder.


- [x] **Bulyan** - The Hidden Vulnerability of Distributed Learning in Byzantium [[ICML 2018]](https://proceedings.mlr.press/v80/mhamdi18a.html)
- [x] **Centered Clipping** - Learning from history for Byzantine robust optimization [[ICML 2021]](http://proceedings.mlr.press/v139/karimireddy21a.html)
- [x] **Centered Median** - Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates [[ICML 2018]](http://proceedings.mlr.press/v80/yin18a/yin18a.pdf)
- [x] **Krum**  - Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent [[Neurips 2017]](https://proceedings.neurips.cc/paper_files/paper/2017/file/f4b9ec30ad9f68f89b29639786cb62ef-Paper.pdf)
- [x] **Trimmed Mean** - Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates [[ICML 2018]](http://proceedings.mlr.press/v80/yin18a/yin18a.pdf)
- [x] **SignSGD** - signSGD with Majority Vote is Communication Efficient and Fault Tolerant [[ICLR 2019]](https://openreview.net/pdf?id=BJxhijAcY7)
- [x] **RFA** - Robust Aggregation for Federated Learning [[IEEE 2022 TSP]](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9721118)
- [x] **Sequantial Centered Clipping** -  Byzantines Can Also Learn From History: Fall of Centered Clipping in Federated Learning [[IEEE 2024 TIFS]](https://ieeexplore.ieee.org/document/9636827)
- [x] **FL-Trust** - FLTrust: Byzantine-robust Federated Learning via Trust Bootstrapping [[NDSS 2021]](https://www.ndss-symposium.org/wp-content/uploads/ndss2021_6C-2_24434_paper.pdf)
- [x] **GAS (Krum and Bulyan)** - Byzantine-robust learning on heterogeneous data via gradient splitting} [[ICML 2023]](https://proceedings.mlr.press/v202/liu23d/liu23d.pdf)
- [x] **FedAVG** - [[AISTATS 2016]](http://proceedings.mlr.press/v51/mcmahan16.pdf)


## Byzantine Attacks:
- Attacks can be extended by adding the attack in the `attacks` folder.


- [x] **Label-Flip** - Poisoning Attacks against Support Vector Machines [[ICML 2012]](https://icml.cc/2012/papers/880.pdf)
- [x] **Bit-Flip** - 
- [x] **Gaussian noise** - 
- [x] **Untargeted C&W ()** - Towards evaluating the robustness of neural networks  [[IEEE S&P 2017]](https://ieeexplore.ieee.org/iel7/7957740/7958557/07958570.pdf)
- [x] **Little is enough (ALIE)** - A Little Is Enough: Circumventing Defenses For Distributed Learning [[Neurips]](https://proceedings.neurips.cc/paper_files/paper/2019/file/ec1c59141046cd1866bbbcdfb6ae31d4-Paper.pdf)
- [x] **Inner product Manipulation (IPM)** - Fall of Empires: Breaking Byzantine-tolerant SGD by Inner Product Manipulation [[UAI 2019]](http://auai.org/uai2019/proceedings/papers/83.pdf)
- [x] **Relocated orthogonal perturbation (ROP)** - Byzantines Can Also Learn From History: Fall of Centered Clipping in Federated Learning [[IEEE 2024 TIFS]](https://ieeexplore.ieee.org/document/9636827)
- [x] **Min-sum** - Manipulating the byzantine: Optimizing model poisoning attacks and defenses for federated learning [[NDSS 2022]] (https://par.nsf.gov/servlets/purl/10286354)
- [x] **Min-max** - Manipulating the byzantine: Optimizing model poisoning attacks and defenses for federated learning [[NDSS 2022]] (https://par.nsf.gov/servlets/purl/10286354)
- [x] **Sparse** - Aggressive or Imperceptible, or Both: Network Pruning Assisted Hybrid Byzantines in Federated Learning
- [x] **Sparse-Optimized** - Aggressive or Imperceptible, or Both: Network Pruning Assisted Hybrid Byzantines in Federated Learning


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

## Future models:
- [x] **Visual Transformers** (ViT , DeiT, Swin, Twin, etc.) 


## Installation

1. Install Python 3.8. For convenience, execute the following command.

```
pip install -r requirements.txt
```

## Citation

If you find this repo useful, please cite our papers.

```
@ARTICLE{ROP,
  author={Ozfatura, Kerem and Ozfatura, Emre and Kupcu, Alptekin and Gunduz, Deniz},
  journal={IEEE Transactions on Information Forensics and Security}, 
  title={Byzantines Can Also Learn From History: Fall of Centered Clipping in Federated Learning}, 
  year={2024},
  volume={19},
  number={},
  pages={2010-2022},
  doi={10.1109/TIFS.2023.3345171}}
```

```
@misc{sparseATK,
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
