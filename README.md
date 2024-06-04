# Subhomogeneous Deep Equilibrium Models (SubDEQ)

Code release for **Subhomogeneous Deep Equilibrium Models (ICML 2024)**.

[[paper]](https://arxiv.org/pdf/2403.00720)

## Requirements
Can install with `pip install -r requirements.txt`.

## Getting started
run `main.py` to re-do our experiments on MNIST, CIFAR100, SVHN, and Tiny ImageNe using the SubDEQ.

In the appnp folder run `train.py` to re-do our experiments on Cora citation, Cora author, CiteSeer, DBLP, and PubMed using the SubDEQ GNN.

## Acknowledgement
The implementation of SubDEQ is based on [Deep Implicit Layers - Neural ODEs, Deep Equilibirum Models, and Beyond](https://implicit-layers-tutorial.org/).

The implementation of APNNP is based on [A PyTorch implementation of Predict then Propagate: Graph Neural Networks meet Personalized PageRank](https://github.com/benedekrozemberczki/APPNP)

## Citation
If you find this repository useful in your research, please consider citing:

```
@inproceedings{sittoni2024subhomogeneous,
  title = {Subhomogeneous Deep Equilibrium Models},
  author = {Pietro Sittoni and Tudisco, Francesco},
  booktitle={International Conference on Machine Learning (ICML)},
  year = {2024}
}
