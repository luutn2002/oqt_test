# Testing git for qst-nn

Quantum state classification and reconstruction with deep neural networks. This repository contains the code to experiment the results presented in the paper: 


> *Classification and reconstruction of optical quantum states with deep neural networks.
Shahnawaz Ahmed, Carlos Sánchez Muñoz, Franco Nori, and Anton Frisk Kockum
Phys. Rev. Research 3, 033278 – Published 27 September 2021
arXiv: [https://arxiv.org/abs/2012.02185](https://arxiv.org/abs/2012.02185)*

## Installation
If you don't have python3.8:
```
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install python3.8
sudo apt-get install python3.8-venv
```
Create environment with:
```
python3.8 -m venv /path/to/new/virtual/environment
```
Setup environment with:
```
pip install -r /path/to/requirements.txt
```
Finally, run:
```
python setup.py develop
```
