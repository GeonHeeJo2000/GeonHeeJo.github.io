---
layout: post
title: Train longer, generalize better
subtitle:  “Train longer, generalize better:” closing the generalization gap in large batch training of neural networks
---

위 논문은 **Neural Information Processing Systems (NeurIPS) 2017**에 출판되었는데, 대규모 배치 학습에서 발생하는 일반화 격차를 줄이기 위한 방법을 제안하며 학습률과 학습 방법을 조정하여 성능을 향상시키는 전략을 탐구합니다.

### Abstract
- 대규모 배치 크기(Large Batch Size)를 사용할 때 generalization performance이 떨어지는 **일반화 격차(generalization gap)** 문제가 발생하는데, 이를 해결하기 위해 다양한 연구들이 진행되어 왔습니다.
- 본 논문에서는 이러한 **generalization gap**가 충분한 학습이 이루어지지 않았기 때문에 발생한다고 주장합니다. 따라서, 학습 체계를 조정하면 이 문제를 해결할 수 있다고 제안합니다.
- 특히, 초기에는 높은 학습률(learning rate)을 사용하고, 추후에는 **초기 가중치와의 거리**를 바탕으로 학습률을 조정함으로써 일반적으로 사용되는 검증 오류(validation error)를 기반으로 한 학습률 조정 방식과는 다른 접근법을 제시합니다.

## Introduction
-  딥 뉴럴 네트워크(DNN, Deep Neural Network)는 매우 complex하고 non-convex하기 때문에 optimization method로는 주로 **SGD(Stochastic Gradient Descent)**를 사용한다.
-  SGD를 사용하여 모델의 generalization performance이 높게 나타나는 현상을 설명하는 연구([Understanding deep learning requires rethinking generalization](https://arxiv.org/abs/1611.03530))도 존재한다.
  
    <p align="center">
      <img src="../assets/img/Impact of batch size on classification error.JPG">
      <br>
      Figure 1: Impact of batch size on classification error
    </p> 
    
    - 그림을 보면 batch size가 크면 클수록 validation error가 증가하는 것을 확인할 수 있다. 이러한 현상을 우리는 일반화가 잘 되지 않았다고 하며, 일반적으로 batch size가 클수록 이러한 현상이 발생한다.
    - 작은 batch size를 사용할 때는 각 배치가 데이터의 일부 샘플만을 사용하여 업데이트되기 때문에 샘플의 다양성에 의한 noise가 발생한다. 이러한 노이즈는 모델이 다양한 parameter space를 탐색하도록 하여 더 넓고 평탄한 최소점(flat minima)을 찾을 수 있다. 그러나 큰 batch size를 사용할 경우, 각 배치가 데이터 전체에 대한 평균을 사용하기 때문에 노이즈가 줄어들고 이로 인해 local minima에 빠질 수 있다.
