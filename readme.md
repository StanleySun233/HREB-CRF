# HREB-CRF: Hierarchical Reduced-bias EMA for Chinese Named Entity Recognition

This work is accepted by [IJCNN2025](https://2025.ijcnn.org/).

Preprint is available at [![arXiv](https://img.shields.io/badge/arXiv-2503.01217-b31b1b.svg)](https://arxiv.org/abs/2503.01217).

## Introduction
HREB-CRF is a deep learning framework for Chinese named entity recognition (CNER). The framework effectively solves the problems of boundary demarcation errors, complex semantic representation, and sound-meaning differences in Chinese named entity recognition through a hierarchical exponential moving average (EMA) mechanism.

Main features:
- Adopting a hierarchical exponential moving average mechanism to enhance word boundary recognition
- Optimizing long text gradient pooling through exponential fixed bias weighted average of local and global hierarchical attention
- Significant improvements have been achieved on the MSRA, Resume, and Weibo datasets, with F1 scores increased by 1.1%, 1.6%, and 9.8%.

## Env Setup
This environment setup is valid in Window10 (GTX2060) and Ubuntu22.04 (GTX4090).

Use `PyTorch=2.3.0` with `CUDA=11.8` or replace your own pytorch version.

```shell
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu118
```

Other Module
```shell
pip install jupyter notebook
pip install datasets scikit-learn transformers==4.41.0 mega_pytorch pytorch-crf
pip install accelerate -U
```

Then, start with notebook [HREB-CRF.ipynb](HREB-CRF.ipynb) with train and evaluate.

The model structure is shown at [model.py](model.py).

## Cite
If this work is help to yours, please cite us!!!

```bib
@article{sun2025hreb,
  title={HREB-CRF: Hierarchical Reduced-bias EMA for Chinese Named Entity Recognition},
  author={Sun, Sijin and Deng, Ming and Yu, Xinrui and Zhao, Liangbin},
  journal={arXiv preprint arXiv:2503.01217},
  year={2025}
}
```
