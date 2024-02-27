    ---
layout: post
title: Not All Passes Are Created Equal
subtitle:  “Not All Passes Are Created Equal:” Objectively Measuring the Risk and Reward of Passes in Soccer from Tracking Data
---

이것은 2017년 KDD Applied Data Science Paper로 제출한 논문으로 축구에서 패스의 가치를 평가하는 논문입니다. 이번 블로그에서는 위 논문에 대해서 자세히 설명하고자 합니다.

- 이 논문은 2017년에 발표되었으며, 기술적으로 어려운 문제를 다루고 있지는 않습니다. 연구에서는 이진 분류 문제를 해결하기 위해 logistic regressor 모델을 사용하고 있습니다. 축구 데이터 분석이 초기 단계에 있었던 시점에 발표된 이 논문은, 기술적인 어려움보다는 축구 데이터 분석에 접근하는 방법을 다루기에 적합한 논문이라고 할 수 있습니다.

### Abstract
- 본 논문은 패스의 가치를 평가하는 새로운 방식을 제안합니다.
- 과거 : 패스를 평가할 때 사람이 직접 annotation를 달아야하기 때문에 단순한 방식으로 패스를 성공했는지 실패했는지로만 평가를 했습니다.
- 논문 : 논문은 패스의 가치를 평가할 때, 단순한 binary value가 아닌 continuous specturum으로 측정해야 한다고 주장합니다. 뿐만 아니라 여러 관점에서 패스의 가치를 평가해야한다고 생각합니다. 그래서 본 논문은 패스 성공 확률과 패스가 chance를 만들 수 있는 확률관점에서 패스의 가치를 평가하고자 합니다.

    ![Figure1](../assets/img/figure1.jpg)
      
    - Figure1은 축구 경기 상황에서 두 가지 다른 패스 선택의 예를 보여주고 있다.
      
        - 왼쪽 사진은 MATIC가 FABREGAS에게 패스하는 상황이고, 오른쪽 사진은 MATIC가 COSTA에게 패스하는 상황이다. 어느 패스가 더 가치있다고 생각하나요?
        - 우리는 오른쪽 사진이 더 위험하지만, 성공을 한다면 더 높은 shooting chance를 만들 수 있는 패스이다. 그만큼 파브레가스한테 패스하는 오른쪽 상황보다 너 많은 스킬이 필요합니다. 그러나 현재 패스 지표(binary value)에서는 두 상황의 패스 모두 같은 가중치를 갖고 있습니다. 이는 게임 상황을 반영하지 않고 선수과 팀의 지표에 영향을 미칠 수도 있다.
        - 본 연구에서는 더 나은 대안으로 risk(패스 성공 확률)과 reward(goal로 이어질 확률)을 고려해야한다고 주장합니다.

    ![Figure2](../assets/img/figure2.jpg)
  
    - Figure2은 risk과 reward를 고려한 두 가지 다른 패스 선택의 예를 보여주고 있다.
      
        - COSTA에게 패스는 성공확률이 40%로 낮지만, 슛으로 이어질 확률은 31%가 증가한다.
        * 여기서 왜 31%인지는 필자도 모르겠다. 이전 shot danger이 4%이고, COSTA에게 패스할 때 shot danger이 33%이면 29%가 증가한거 아닌가?
        - 이러한 risk과 reward를 객관적으로 추정하는 것을 보여줄 예정이다.

### DataSet
- 본 논문에서는 0.1초마다 수집되는 위치정보가 포함된 trackingd-data과 event-name, the ball location, possession등의 이벤트 관련 정보가 들어있는 event-data를 활용했다. 수집한 데이터는 2014/2015~2015/2016 season EPL(English Premier League)의 726경기를 가져왔다.
- 726경기에서 발생한 총 패스는 571,287개이고, 이 중 패스가 성공한 횟수는 468,265개이다. 경기 당 패스의 수로 비교했을 때는, 평균적으로 380.46개가 발생하고 그 중 320.91개 성공했다. 저희의 baseline으로 사용한 패스성공확률은 84.35%이다.
  
    ![Table1](../assets/img/table1.jpg)
  
    - Table1은 패스 이벤트의 summary이다.
      
        - 백패스 or 사이드패스에 비해 전방패스의 성공확률이 훨씬 낮다.
        - 경기장을 3등분 했을 때, 우리팀 진영이 first third이고 상대팀 진영이 final third이다. 상대팀 진영과 가까운 공간에서 패스할 수록 패스 성공확률이 더 낮은 것을 볼 수 있다.
        - 패스하는 위치 or 패스의 종류에 따라서 성공확률이 달라지는 것을 확인할 수 있었다.
          * 우리는 이러한 패스의 context정보도 risk를 제대로 측정하는데 도움을 줄 것이다.
          
### AE(AutoEncoder)
- 오토인코더(AE)는 입력 데이터를 압축한 후 복원하여 representation learning(데이터의 표현을 학습)하는 비지도 학습 알고리즘이다.
  
      1. Encoder : 입력 데이터를 내부 표현(잠재 공간)으로 변환 -> 추출된 특징을 Latent Vector라고 부름
  
      2. Decoder : Encoder를 거친 Latent Space를 받아 원본 데이터과 같은 형태로 재구성
  
- Model code github : [Model](https://github.com/dariocazzani/pytorch-AE/blob/master/models/AE.py)
  
![Model](https://blog.kakaocdn.net/dn/8JonH/btqFBec9cAF/mhxdDF930R0CrHs9NdUKv1/img.png)
  
### VAE(Variational AutoEncoder)
- 변이형 오토인코더(VAE)는 AE과 비슷한 구조를 가지지만, 확률 분포를 모델링한다는 점에서 차이가 있다

      1. Encoder : 입력 데이터를 내부 표현(잠재 공간)으로 변환 -> 확률 분포를 정의하는 평균과 표준편차 출력
  
      2. Latent Space : AE과 다르게 VAE에서는 Latent Space에서 noise를 추가함(동일한 데이터 생성 방지)
      * 정규분포로 부터 하나의 noise를 샘플링한 후 이를 바탕으로 Latent vector z를 얻는데, 이를 reparameterize이라 한다.
  
      3. Decoder : Latend Space를 거친 z를 받아 원본 데이터과 같은 형태로 재구성
  
- Model code github : [Model](https://github.com/dariocazzani/pytorch-AE/blob/master/models/VAE.py)
  
![Model](https://blog.kakaocdn.net/dn/b30Uzl/btrxY4wKngj/SucVwitDrRtQvi1xTHdrR0/img.png)

### VRNN
- RNN의 시간적 동적 특성과 VAE의 확률적 생성 모델링를 결합했다. 시간에 따라 변화하는 Trajectory를 효과적으로 학습하기 위해서 RNN도입
  
      1. Prior : 데이터를 접근하기 전 가지고 있는 사전 분포를 통해서 데이터를 추정함.
          * Encoder가 입력데이터를 받아 Latent Space표현으로 변환하는 역할을 한다면, Prior은 Latent Space에 대한 전체적인 구조            와 분포를 정의함으로써 데이터를 생성할 때 일반화능력을 향상시킬 수 있다.
          * 수식은 t시점 이전(과거)의 정보만을 활용한 분포를 추정하는 식임을 확인할 수 있다.
  
  $$\ \text{p}_{\theta}(z_t | x_{<t}, z_{<t}) = \ \phi_{\text{prior}}(h_{t-1})$$

      2. Latent Space : VAE과 같은 역할이다. 학습할 때는 Encoder의 확률 분포를 받고, 데이터 생성할 때는 Prior의 확률분포를 받는다.

      3. Encoder(Inference) : 입력 데이터를 내부 표현(잠재 공간)으로 변환
      * 학습할 때 사용
      
  $$\ \text{q}_{\theta}(z_t | x_{\leq t}, z_{<t}) = \ \phi_{\text{enc}}(x_t,h_{t-1})$$

      4. Decoder(Generation) : Latend Space를 거친 z를 받아 원본 데이터과 같은 형태로 재구성
      
  $$\ \text{p}_{\theta}(z_t | z_{\leq t}, x_{<t}) = \ \phi_{\text{dec}}(z_t,h_{t-1})$$

      5. Recurrence : 이전 시점의 hidden state과 입력데이터, Latent Vecor를 활용하여 현재 시점의 hidden state를 업데이터하는 과정이다
      
  $$\ \text{h}_t = f(x_t, z_t, h_{t-1})$$
  

- Loss : VAE의 손실 함수는 주로 두 부분으로 구성됩니다: 재구성 손실(reconstruction loss)과 정규화 손실(Kullback-Leibler divergence). 이 두 요소를 합쳐 Evidence Lower Bound (ELBO)라고 하며, VAE의 목표는 ELBO를 최대화하는 것입니다.
  
      1. Reconstruction Loss
            - 입력 데이터와 출력 데이터 간의 차이를 줄입니다. 모델이 데이터를 얼마나 잘 재구성하는지 측정하는 지표
  
      2. Regularization Loss
            - 모델이 단순히 데이터를 재생산하는 것을 넘어서, 일반적인 데이터 패턴을 이해하고 새로운 데이터를 생성할 수 있게 하는 정규화 역할(입력 데이터의 복사본 생성 방지)
            - KL divergence는 Trajectory관점에서 다음 위치를 예측할 때, 이전 위치와 동일한 위치를 생성하지 않도록 하기 위한 것과 유사합니다. 즉, 모델이 데이터의 다양성을 유지하고 예측 가능한 패턴을 학습하도록 유도합니다.

- Model code link : [Model](https://github.com/emited/VariationalRecurrentNeuralNetwork/blob/master/model.py)
  
![image](https://github.com/GunHeeJoe/GunHeeJoe.github.io/assets/112679136/19c89399-9ba1-463e-8867-ea61078dec90)


  

### GVRNN
- GVRNN은 VRNN에 GNN기법을 추가한 것이다. 축구의 경우 각 선수들의 Trajectory는 다른 선수들의 영향을 받기 때문에 GNN기법도 추가한 것이다.
* 다중 에이전트의 상호작용을 학습
- Prior, Encoder(Inference), Decoder(Generation)를 통해 나온 확률분포에 GNN를 추가하므로써 선수들의 상호작용도 학습하는 구조
- Model code link : [Model](https://github.com/keisuke198619/C-OBSO/blob/main/vrnn/models/gvrnn.py)
  
![Model](https://github.com/GunHeeJoe/GunHeeJoe.github.io/assets/112679136/605202a0-3cf4-422e-87f5-fa1f8932cfcb)


