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
  
    <p align="center">
      <img src="../assets/img/figure1.jpg">
      <br>
      Figure1
    </p>
      
    - Figure1은 축구 경기 상황에서 두 가지 다른 패스 선택의 예를 보여주고 있다.
      
        - 왼쪽 사진은 MATIC가 FABREGAS에게 패스하는 상황이고, 오른쪽 사진은 MATIC가 COSTA에게 패스하는 상황이다. 어느 패스가 더 가치있다고 생각하나요?
        - 우리는 오른쪽 사진이 더 위험하지만, 성공을 한다면 더 높은 shooting chance를 만들 수 있는 패스이다. 그만큼 파브레가스한테 패스하는 오른쪽 상황보다 너 많은 스킬이 필요합니다. 그러나 현재 패스 지표(binary value)에서는 두 상황의 패스 모두 같은 가중치를 갖고 있습니다. 이는 게임 상황을 반영하지 않고 선수과 팀의 지표에 영향을 미칠 수도 있다.
        - 본 연구에서는 더 나은 대안으로 Risk(패스 성공 확률)과 Reward(goal로 이어질 확률)을 고려해야한다고 주장합니다.
 
    <p align="center">
      <img src="../assets/img/figure2.jpg">
      <br>
      Figure2
    </p>
    
    - Figure2은 Risk과 Reward를 고려한 두 가지 다른 패스 선택의 예를 보여주고 있다.
      
        - COSTA에게 패스는 성공확률이 40%로 낮지만, 슛으로 이어질 확률은 31%가 증가한다.
        - 여기서 왜 31%인지는 필자도 모르겠다. 이전 shot danger이 4%이고, COSTA에게 패스할 때 shot danger이 33%이면 29%가 증가한거 아닌가?
        - 이러한 Risk과 Reward를 객관적으로 추정하는 것을 보여줄 예정이다.

## DataSet
- 본 논문에서는 0.1초마다 수집되는 위치정보가 포함된 trackingd-data과 event-name, the ball location, possession등의 이벤트 관련 정보가 들어있는 event-data를 활용했다. 수집한 데이터는 2014/2015~2015/2016 season EPL(English Premier League)의 726경기를 가져왔다.
- 726경기에서 발생한 총 패스는 571,287개이고, 이 중 패스가 성공한 횟수는 468,265개이다. 경기 당 패스의 수로 비교했을 때는, 평균적으로 380.46개가 발생하고 그 중 320.91개 성공했다. 저희의 baseline으로 사용한 패스성공확률은 84.35%이다.

    <p align="center">
      <img src="../assets/img/table1.jpg">
      <br>
      Table1
    </p>
   
    - Table1은 패스 이벤트의 summary이다.
      
        - 백패스 or 사이드패스에 비해 전방패스의 성공확률이 훨씬 낮다.
        - 경기장을 3등분 했을 때, 우리팀 진영이 first third이고 상대팀 진영이 final third이다. 상대팀 진영과 가까운 공간에서 패스할 수록 패스 성공확률이 더 낮은 것을 볼 수 있다.
        - 패스하는 위치 or 패스의 종류에 따라서 성공확률이 달라지는 것을 확인할 수 있음 -> 패스의 context정보도 Risk를 측정하는데 활용
          
## Risk and Reward

|   |Risk|Reward|
|:---:|:---:|:---:|
|설명|The likelihood of successfully executing a pass|The likelihood of a pass creating a shot chance|
|Label|The outcome of pass event|1 if a shot is taken within 10 seconds after a pass, otherwise 0|


## Model
- Random Forest : 비선형 블랙박스 모델로 예측 과정을 해석하기 어려움. -> coaching point(coefficient)가 필요함.
- Logistic Regressor : 패스의 성능에 영향을 주는 요인을 코치들도 분석하는 것이 중요하므로 본 연구에서는 선형 모델인 Logistic Regerssor를 활용한다.
- Data : Train(352,466) & Valid(114,257) & Test(114,257)


## Context Features
- 패스 성공률이 context과 관계가 있다는 것은 table1에서 확인할 수 있었다. 앞서봤던 micro-level뿐 아니라 더 높은 수준의 contextual infromation이 예측을 향상시키고 코치들에게 유용한 정보를 제공할지도 모른다.

    <p align="center">
      <img src="https://d3i71xaburhd42.cloudfront.net/3bc06b64581287361771ca4bb95f74991abb805d/5-Figure5-1.png">
      <br>
      Figure5
    </p>

    - Figure5은 패스의 contextual feature를 담은 passing dictionary이다.
        - 우리는 3가지 구조의 contextual feature를 만들어서 예측 성능을 향상시키고자한다.

      <br>
      
      **1. Micro Feature**
    
        - 대표적인 패스의 context정보 : 속도, 거리, 패스각도, 첫터치 시간등을 사용한다.
        - 소유권을 되찾은 후 시간 : 상대팀이 조직을 갖춘 상황인지 이미 조직을 갖춘 상황인지에 따라 패스성공확률은 달라진다.
        - Expected Receiver(Intented Recevier) : 패스의 의도된 receiver
  
        <p align="center">
          Expected Receiver = \(\frac{\text{Distance}}{\text{Min Distance}} \times \frac{\text{Angle}}{\text{Min Angle}}\)
        </p>
  
        - 패스의 성공 확률은 패스 수신자의 개인적인 기술에 영향을 미친다. 롱패스를 잘 받기로 유명한 Didier Drogba에게 패스를 한다면, 패스스킬이 좋지 않은 선수가 패스를 하거나 받기 어려운 패스도 패스 성공 확률이 높아질 수 있다.
        - 그러나, 실패한 패스의 경우 실제로 수신자가 누구인지 알 수 없기 때문에, 패스 수신자의 개인적인 기술을 고려할 수 없다. 이 때, 고안해낸 것이 Expected Receiver(Intended Receiver)이다.
        - 이전에 발표한 "Beyond Completion Rate: Evaluating the Passing Ability of Footballers"라는 논문은 실패한 패스 위치에서 가장 가까운 수신자를 Intended Receiver라고 정의했지만, 본 연구에서는 잠재적인 수신자들의 각도까지도 고려하여 Intended Receiver를 정의하고자 한다.
        - 그러나, Intended Receiver에도 단점이 존재한다. 실패한 패스 위치 주위에 사람이 여러명 있거나 패스가 초기에 차단당했을 경우 Intended Receiver를 예측하기는 어렵다. 뿐만 아니라 패스가 실제로 달리는 선수 앞으로 떨어졌는지 뒤로 떨어졌는지 알 수 없기 때문에 패스의 절대적인 위치만을 활용하는 것이. Intended Receiver의 한계이다.
        * 아직까지 Intended Receiver를 정확하게 분류하는 연구를 많이 보지는 못했다. 실제로도 제가 봤을 때는 거의 없었다. 그나마 가장 좋았던 것이 Expected pass라는 논문에서 Intended Receiver를 예측하는 연구를 했었는데, 성공한 패스는 93%, 실패한 패스는 72%로 나왔었다.
  
      **2. Tactical Feature**
    
        - open-play(세트피스 상황과 같이 멈춰있는 상황이 아닌 경기가 진행되고 있는 상황을 의미)ㅇ에 집중하여 3가지 game-state로 분류한다 : build-up, counter-attack, unstructed-play
        - Tactical Feature는 context를 더 용이하게 분석할 뿐 아니라 Risk과 Reward를 향상시킬 것으로 기대함
        * 2024-02-07에 한국 vs 요르단경기도 이것을 활용하면 counter-attack상황에서 한국의 패스 성공률이 어떻게 측정될지도 궁금하네요.

      **3. Formation Feature**

        - 패스의 Risk는 defensive-block에도 영향을 미친다. 따라서 우리는 high-block, medium-block, low-block으로 분류한다
        - defensive-block를 사용하기 위해서 선수과 공의 위치좌표를 클러스터링을 사용했다.
     
        - contextual information를 추가적으로 사용하기 위해 formation clustering도 활용한다.
        - formation clustering은 모든 선수들의 위치 정보를 활용하여 비슷한 formation정보를 찾아주는 논문입니다.
        - 이를 활용하면 패스가 수비수과 미드필더 사이에서 패스를 했는지, 미드필더과 공격수 사이에서 패스를 했는지를 파악할 수 있다.
      
        <p align="center">
              <img src="https://d3i71xaburhd42.cloudfront.net/34b4f2ae4d541be465ee34a6d168d80edd18123e/4-Figure5-1.png">
              <br>
              Formation Clustering
            </p>

        - 위 그림은 "Large-scale analysis of soccer matches using spatiotemporal tracking data"에서 제안한 Formation clstering했을 때 나온 유사한 formation그림이다.

    <p align="center">
      <img src="https://d3i71xaburhd42.cloudfront.net/3bc06b64581287361771ca4bb95f74991abb805d/5-Figure4-1.png">
      <br>
      Figure4
    </p>
   
    - Figure4은 tactical and formation features을 보여준다. 노란색 선은 오른쪽 상단에 Pass Risk(패스 성공 확률)과 Pass Reward를 표현한 패스이다.
    <br>
    
    |   |Left|Right|
    |:---:|:---:|:---:|
    |상황|골키퍼가 공을 잡자마자 빠르게 진행하는 역습 상황|상대 수비 조직을 무너트리기 위한 빌드업 단계|
    |Feature|역습 상황인 counter-attack모습과 defensive-block를 갖추지 못한 high-block상황을 보여줌|Build-up모습과 defensive-block를 갖춘 low-block상황을 보여줌|
    |**risk**|수비수들이 조직을 갖추지 않은 상태에서 패스가 대부분이므로 평균적으로 risk가 낮음|빌드업단계에서 패스는 risk가 낮고, 침투패스는 risk가 높음|
    |Reward(danger)|진취적인 성격을 띄는 패스가 많으므로 평균적으로 reward(danger)가 높은 것을 확인함|측면패스가 많으므로 평균적으로 reward(danger)이 낮음| 

    * Risk : 패스 성공 확률 -> Risk가 높으면 패스 성공 확률이 높다
    * **risk(위험도)** : 패스를 성공적으로 완료하는데 있어서 위험성 -> risk가 낮으면, 안전한 패스이다.
    * danger(위험성) : Reward과 같은 의미 -> Danger가 높으면, 슛팅할 확률이 높다.
    <br>
    
    * 진취적인 패스(Progressive Pass) : 하프라인을 기준으로 우리팀 진영에서 30m이상의 패스 or 상대팀 진영에서 10m이상의 패스
    * 진취적인 패스 관련 기사 : [Progressive Pass](https://www.interfootball.co.kr/news/articleView.html?idxno=381033)

        
## Match Analysis
- 이제 본 연구에서 제안하는 Pass Risk과 Pass Reward를 활용하여 실제 경기에서 어떻게 분석할 수 있는지 확인해보자.
  
    <p align="left">
          <img src="https://d3i71xaburhd42.cloudfront.net/3bc06b64581287361771ca4bb95f74991abb805d/250px/5-Figure6-1.png">
          <br>
          Figure 6
        </p>
    <p align="right">
          <img src="https://d3i71xaburhd42.cloudfront.net/3bc06b64581287361771ca4bb95f74991abb805d/250px/6-Figure7-1.png">
          <br>
          Figure 7
        </p>

## VRNN
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
