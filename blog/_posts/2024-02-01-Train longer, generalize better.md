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
      - 작은 batch size를 사용할 때는 각 배치가 데이터의 일부 샘플만을 사용하여 업데이트되기 때문에 샘플의 다양성에 의한 noise가 발생한다. 이러한 노이즈는 모델이 다양한 parameter space를 탐색하도록 하여 더 넓고 평탄한 최소점(flat minima)을 찾을 수 있다. 그러나 큰 batch size를 사용할 경우, 각 배치가 데이터 전체에 대한 평균을 사용하기 때문에 노이즈가 줄어들고 이로 인해 local minima에 빠질 수 있다

#### Why Should We Use Large Batch Sizes?
1. Parallelize : 병렬 연산 향상
2. Learning time : 학습 수렴 속도 향상
3. Memory : 메모리와 자원을 효율성 향상

## Contribution
- 본 논문은 Large Batch Size를 사용할 때 발생하는 generalization gap 문제를 다양한 학습 체계를 활용하여 해결하고자 한다. 저자가 강조하는 핵심 메시지는 다음 문장에 담겨 있다: 
  - **"There is no inherent 'generalization gap': large-batch training can generalize as well as small batch training by adapting the number of iterations."**
  - 결국, 일반화 격차는 large batch size 자체에서 비롯된 것이 아니며, 충분한 학습 반복 횟수와 학습 체계를 조정하면 해결될 수 있다는 것을 말하고 있다.
- Generalization gap을 해결하기 위해 이 논문에서는 **learning rate**과 **batch normalization**를 조정했다. 특히, 일반적으로 사용되는 training or validation errors의 변화에 따른 learning rate 조정 없이, 초기 높은 학습률을 사용했다.


## Method
- 이론적인 부분은 깊게 다루지 않는다.
  
    **3. Theoretical Analysis**
    - SGD(Stochastic Gradient Descent) 기반의 딥러닝 최적화 과정을 수학적으로 설명한 부분이기 때문에 자세히 다루지는 않는다.

    **3. Model: Random Walk on a Random Potential**
    - 딥러닝에서 optimization 과정을 통계 물리학 관점에서 분석한 내용인데, 이론적인 부분은 모두 알지 못하지만 기본적인 내용은 파악하고 넘어간다.
    - 결국 저자가 말하고 싶은 것은 DNN loss surface가 랜덤 워크(불규칙하게 움직이는 입자의 경로를 설명하는 확률적 과정)나 랜덤 포텐셜(입자가 무작위의 힘에 의해 움직이는 환경)처럼 복잡한 형태를 가지고 있다는 것이다.
    - 본 논문에서는 딥러닝의 입자 움직임이 "ultra-slow diffusion"이라고 주장한다. 수식으로 설명하면, 입자가 이동한 거리가 다음과 같이 $$\log t$$ 형태로 증가한다는 것이다:

    $$
    \|\mathbf{w}_t - \mathbf{w}_0\| \sim \log t
    $$

    <p align="center">
      <img src="../assets/img/Euclidean distance of weight vector from initialization.JPG">
      <br>
      Figure 2: Euclidean distance of weight vector from initialization
    </p> 

      - 위 그림은 학습 시간이 경과함에 따라 초기 가중치 벡터와의 거리를 시각화한 것이다.
      - 결론적으로 저자들은 딥러닝 모델이 손실 함수 공간에서 최적의 위치를 찾기 위해 많은 시간이 소요된다고 주장한다.
    
