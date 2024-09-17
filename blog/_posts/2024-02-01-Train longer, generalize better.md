---
layout: post
title: Train longer, generalize better
subtitle: Train longer, generalize better: closing the generalization gap in large batch training of neural networks
---

위 논문은 **Neural Information Processing Systems (NeurIPS) 2017**에 출판되었는데, 대규모 배치 학습에서 발생하는 일반화 격차를 줄이기 위한 방법을 제안하며 학습률과 학습 방법을 조정하여 성능을 향상시키는 전략을 탐구합니다.

### Abstract
- 대규모 배치 크기(Large Batch Size)를 사용할 때 generalization performance이 떨어지는 **일반화 격차(generalization gap)** 문제가 발생하는데, 이를 해결하기 위해 다양한 연구들이 진행되어 왔습니다.
- 본 논문에서는 이러한 **generalization gap**가 충분한 학습이 이루어지지 않았기 때문에 발생한다고 주장합니다. 따라서, 학습 체계를 조정하면 이 문제를 해결할 수 있다고 제안합니다.
- 특히, 초기에는 높은 학습률(learning rate)을 사용하고, 추후에는 **초기 가중치와의 거리**를 바탕으로 학습률을 조정함으로써 일반적으로 사용되는 검증 오류(validation error)를 기반으로 한 학습률 조정 방식과는 다른 접근법을 제시합니다.

## Introduction
- DNN(Deep Neural Network)은 매우 complex하고 non-convex하기 때문에 optimization method로는 SGD(Stocahstic gradient decent)를 활용한다. 그리고 SGD를 활용하여 과적합되지 않고 일반화가 잘 되고 이를 설명하는 연구([Understanding deep learning requires rethinking generalization](https://arxiv.org/abs/1611.03530))도 존재한다.
