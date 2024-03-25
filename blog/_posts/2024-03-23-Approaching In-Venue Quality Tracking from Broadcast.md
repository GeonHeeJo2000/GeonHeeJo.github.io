---
layout: post
title: Approaching In-Venue Quality Tracking from BroadcastVideo using Generative AI
subtitle: 2024 MIT Sloan Sports Analytics Conference
---

이것은 2024년 MID SLONE에 제출한 논문으로 축구경기중 카메라에 포착되지 않은 선수들을 보간하는 논문입니다. 이번 블로그에서는 위 논문에 대해서 자세히 설명하고자 합니다. 

- 이 논문은 2024년에 발표된 논문으로 다양한 기술들을 설명하고 있습니다. 그러나, 그 기술들에 대해서 자세히 설명하지 않으니 궁금한 것들에 대해서는 따로 공부하는 것을 추천한다.
- AI : GNN(Graph Neural Networks), SAA(Spatiotemporal Axial Attention), Temporal Attention, Spatial Attention, Self-Attention, Diffuion
[Diffuion](https://gunheejoe.github.io/2024-03-22-Diffusion/)

### Abstract
- 축구에서 tracking-data는 25년이 되었고, tracking-data를 활용하여 다양한 분석을 할 수 있었다. tracking-data는 초당 10개의 frame으로 선수의 위치를 추적하는데, 초창기에는 경기장에 설치된 카메라 or human에 의해서 수행되었다. 그리고 이를 활용하여 다양한 축구 데이터 분석을 수행할 수 있었다. 
- 2008년 computer vision에 발전에 힘입어 자동으로 선수과 공의 위치를 추적할 수 있었고, 이에 실시간으로 데이터 분석도 가능해졌다. 그러나, 이러한 tracking-data를 제한된 가용성으로 인해 광범위한 활용에 제약이 있다. 자신의 팀만 사용할 수 있거나, 공유를 한다해도 리그 내에서의 분석만 가능하게 하여 국제적인 분석과 비교를 어렵게 만듭니다. 
<br/> * (b)가 카메라를 통해 자동으로 선수과 공의 위치좌표를 추출한 후에 시각화한 그림이다.
- broadcast tracking system의 발전으로 인해 이러한 제한을 극복할 수 있었다. 그러나, 방송에서 얻은 데이터는 주요 카메라에서 벗어난 선수, 근접 슛 촬영 , 화질, 선수가 선수를 가리는 장면등 여러 원인으로 인해 불완정합니다. 본 연구에서는 이러한 문제를 해결하기 위해 Diffusion Model를 활용해서 카메라에 포착되지 않는 선수들을 보간하고자합니다. 
<br/> * (a)과 (d)가 방송을 통해 선수과 공의 좌표를 추출한 후에 시각화한 그림이다.
  
    <p align="center">
      <img src="../assets/img/inputation-all-data.JPG">
      <br>
      Figure1
    </p>
    
    - (a) : 방송에서 포착된 장면
    - (b) : 실제 경기장의 카메라를 통해 자동으로 위치를 추적한 In-Venue tracking-data
    - (c) : on-the-ball 이벤트만 기록한 event-data
      * event-data로 분석은 가능하지만, off-the-ball은 놓치므로 정확한 분석이 불가능함.
    - (d) : 방송에서 포착된 장면을 통해 선수들의 위치좌표를 추적한 Broadcast tracking-data
    - (e) : 방송 주요 카메라에 포착되지 않은 다른 선수들을 보간한 Imputed tracking-data
      * 본 연구에서는 diffusion를 통해서 보간하고자 한다.  
 
## Model Architecture
- 본 논문에서는 tracking-data를 보간하는 방식은 크게 3가지로 나눈다. 우리는 3가지 방식이 실제로 어떻게 구현되는지 알아볼 예정이다.
1. Encoding Broadcast Tracking Data
2. Encoding Broadcast Event Data
3. generatvie AI model
  
      **1. Encoding Broadcast Tracking Data**

      - tracking-data를 encoding하는 방법은 크게 temporal-attention과 spatial-attention를 활용한다.
      - broadcast tracking-data를 encode하는 것은 겹치는 agents의 위치를 추론하는데 강한 signals를 형성한다.
      - tracking-data를 encoding하는 주요 과제는 (1) modeling each agent's past behaviors과 (2) representing interagent spatial dynamics가 있다. 특히, agents가 오랫동안 겹치기 때문에 어렵다. 따라서 한번에 여러 시간동안의 tracking-data를 encoding해야한다. 
  
     <p align="center">
        <img src="../assets/img/SAA attention.JPG">
        <img src="../assets/img/SAA attention architecture.JPG">
        <br>
        SAA attention
    </p>

    - Temporal Attention과 Spatioal Attention를 연속적으로 처리함 -> SAA(Spatiotemporal axial attention)
      1. Temporal Attention : 각 agent의 과거 위치 간의 self-attention를 계산하여 temporal context를 추출한다. -> 겹치는 agent문제 해결
      2. Spatial Attention : 특정 시점에 모든 agent의 위치 사이의 self-attention를 계산하여 spatial context를 추출한다. -> permutation문제 해결
    
  
      **2. Enhancing Broadcast Tracking with Event Data**

      - tracking-data의 한계 : 공을 지속적으로 추적하는데 어려움 & 짧은 시간동안 방송 추적 제공하지 않음
      - 이러한 기간은 상대적으로 짧지만(<10초) 추가적인 context없이 agent를 합성하는 것은 매우 어렵다.
      - tracking-data과 event-data를 통합하여 이러한 문제를 해결하고자 한다. -> multi-modal, consisting of multiple spatiotemporal input modes 
  
     <p align="center">
        <img src="../assets/img/SAA attention.JPG">\
        <br>
        how we fuse where event data and broadcast tracking-data
    </p>

    - event-data도 spatiotemporal modality로 활용할 수 있음.
      1. Temporal Attention : the chronological ordering of each player’s
events
      2. Spatial Attention :  representing each specific player


      **3. Generating Photorealistic Tracking Data via Diffusion**

      - broadcast tracking-data과 event-data를 결합하면, agent의 위치를 보다 정확하게 예측할 수 있지만 반드시 realistic human motion를 예측하지는 못한다.
      - noise과 heavy occlusions문제로 인해 여전히 위치 정보가 불확실하다. 이는 종종 trajectory가 부드럽지 않고 순간이동하는 것처럼 보인다.
      - 이러한 비현실적인 motion를 해결하기 위해 diffusion을 활용한다.
      
     <p align="center">
        <img src="../assets/img/SAA attention.JPG">\
        <br>
        how we fuse where event data and broadcast tracking-data
    </p>

    - Diffusion과정에 대해서 수식이 존재하지 않아서 정확한 알고리즘은 알 수가 없고, 영상과 논문을 통해 추측해봤다.
       1. human motion이 남을 때까지 noise를 계속 추가한다.
       2. noise sample에서 denoise를 통해 realistic tracking-data를 추출한다.
       3. denoise단계에서 play encoding정보를 전달한다.
       * Diffusion? SoccerDiffusion? LatentDiffusion? 정확한 구조를 알 수 없다. 
   
    **2. Tactical Feature**
  
      - open-play(세트피스 상황과 같이 멈춰있는 상황이 아닌 경기가 진행되고 있는 상황을 의미)에 집중하여 3가지 game-state로 분류한다 : build-up, counter-attack, unstructed-play
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

    <p align="center">
      <img src="../assets/img/figure6.jpg">
      <br>
      Figure8
    </p>

    - Figure 6은 2016년 맨시티(펩 과르디올라) vs 맨유(무리뉴)의 경기에서 맨시티가 0 vs 1으로 패배한 경기에 대한 summary를 보여준다.
        - Basic summary : 점유율(55% vs 45%), 총 패스수(402 vs 266), 패스성공율(88% vs 82%)임을 볼 수 있다. 그러나, 이러한 basic summary를 보면 의문점이 들 수 밖에 없다. 그럼 왜 진거지에 대한 의문을 제기할 수 밖에 없다. 즉, Basic summary만으로는 분석에 한계가 있다.
        - Pass Risk and Pass Reward(danger) : 맨시티는 맨유에 비하여 더 많은 dangerous pass(131 vs 72)를 했고 더 Danger(16% vs 13.5%)를 했지만, Risk(14% vs 17%)는 더 낮았다. 
        - Pass Risk와 Pass Reward(danger)의 관점에서 분석해보면, Pass Risk가 패배의 원인이 될 수 있음을 확인할 수 있습니다. 맨시티는 패스 성공 확률이 낮은 패스를 많이 수행했고, 위험한 패스들을 맨유보다 더 많이 실패했고 이것이 패배의 원인 중 하나이지 않을까 생각이 든다.
        - 실제로, 맨유의 공격 루트 중 상당수가 맨시티의 패스 미스로 인한 역습에서 비롯되었습니다. 이는 경기에서 관찰된 맨시티의 잦은 패스미스가 Pass Risk와 유사함을 확인할 수 있다.
  
    <p align="center">
      <img src="../assets/img/figure7.jpg">
      <br>
      Figure8
    </p>    
    
    - Figure 7은 선수별 Pass Risk와 Pass Reward를 보여준다.
        - Pass Risk가 높은 선수가 주로 골키퍼과 수비수이며, Pass Danger이 높은 선수가 주로 공격수와 공격형 미드필더인 점을 볼 때, Pass Risk과 Pass Reward가 선수들의 특징을 반영하고 있음을 확인할 수 있다. 


## Specific Play Analysis
- 소유권 과정에서 어느 선수가 가장 중요한 패스(Reward가 높은 패스)를 했는지도 파악할 수 있을까? 과연 어시스트를 수행한 선수가 가장 Reward가 높을까?

    <p align="center">
      <img src="https://d3i71xaburhd42.cloudfront.net/3bc06b64581287361771ca4bb95f74991abb805d/6-Figure8-1.png">
      <br>
      Figure8
    </p>

    - Figure 8은 소유권중 각 패스의 Reward를 보여준 그림이다.
        - 실제로 어시스트를 수행한 선수는 Willian이지만, 가장 Reward를 높게 받은 선수는 Fabregas이다. 이는 윌리안의 위험한 지역에서 공을 유지하고 패스하는 능력만 포착하는 것이 아닌 파브레가스의 패스도 Reward가 높다는 것을 식별할 수 있음을 보여준다.

## Application
- 앞서 구한 risk과 reward를 활용하여 여러 지표들을 계산해서 특정 스킬이 뛰어난 선수들을 찾아보자.
  
    **1. PPM(Passing Plus Minus)**
    - 패스의 스킬이 뛰어난 선수는 누구일까?
    - S : Successful -> 어려운 패스를 많이 성공했으면 PPM이 높아진다.
    - U : Unsuccessful -> 쉬운 패스도 실패를 많이하면 PPM은 낮아진다.
  
    <p align="center">
        $$\text{Passing Plus/Minus} = \sum_{s=1}^{S} (1 - y_{risk}^\text{s}) - \sum_{u=1}^{U} (y_{risk}^\text{u} - 1)$$
    </p>
    
    **2. DP(Difficult Pass Completion)**
    - 어려운 패스를 잘 수행하는 선수는 누구일까?
    * 어려운 패스 : risk가 높은 상위 25%의 패스
    - DPS : 어려운 패스의 성공 횟수
    - DPA : 어려운 패스의 총 횟수
      
    <p align="center">
        $$\text{Difficult Pass Completion} = \frac{\sum_{i=1}^{n} \text{i=DPS}}{\sum_{i=1}^{n} \text{i=DPA}}$$
    </p>  
  
    
    **3. PRA(Passes Received Added)**
    - 패스를 잘 받는 선수는 누구일까? 
    - $XD_{pr}$ : 어려운 패스를 잘 받을 확률 -> 패스 성공 확률이 낮은 어려운 패스들을 선수가 많이 받으면, PRA가 높아진다.
  
    <p align="center">
        $$\text{Passes Received Added} = 1 - XD_pr$$
    </p>      
      
    **4. TPA(Total Passes Added)**
    - 모든 것을 고려했을 때, 공 소유에 긍정적인 기여를 하는 선수는 누구일까?
    - TPA = PPM(Passing Plus Minus) + PRA(Passes Received Added)

    <p align="center">
        $$\text{TPA(Total Passes Added} = \text{PPM(Passing Plus Minus)} + \text{PRA(Passes Received Added)}$$
    </p>    
  
    <p align="center">
      <img src="https://d3i71xaburhd42.cloudfront.net/3bc06b64581287361771ca4bb95f74991abb805d/7-Figure9-1.png">
      <br>
      Figure9
    </p>

  - Figure9는 PPM,  DP, PRA, TPA의 관계를 파악하기 위한 그림이다.
  - 왼쪽 그림은 패스의 스킬과 어려운 패스의 스킬이 모두 뛰어난 선수인 외질, 미켈, 파브레가스등을 볼 수 있다. 그러나 전성기에서 한참 지난 맨유의 레전드 루니 선수는 패스 스킬이 매우 낮은 것을 확인할 수 있다
  - 오른쪽 그림은 PPM과 PRA관계 뿐 아니라 원의 크기를 나타낸 TPA도 표현할 수 있었다. 실제로 3가지 지표가 매우 뛰어난 선수인 산체즈 조슈아 킹, 아구에로등을 볼 수 있는데, 이 선수들은 주로 CF(Center Forward)에서 뛰는 선수들이다.
  - 
## TEAM-BASEDANALYSIS
- 각 팀의 패스 스타일 분석하고 어떤 패스 유형이 위협적인지 분석한다.
  
    <p align="center">
      <img src="../assets/img/figure11.jpg">
      <br>
      Figure11
    </p>

  - Figure 11은 패스 출발지와 목적지의 XY좌표를 활용하여 클러스터링한 결과이다.
  - 사각형의 크기는 패스의 빈도수를 색의 강도는 패스의 위험정도를 보여준다.
  - 표를 보면 클러스터링 1,2,3과 같은 패스 스타일은 위협적이지만, 그만큼 risk도 높아서 높은 스킬이 필요하다. 실제로 클러스터링 1,2,3과 같은 패스는 강팀인 아스날과 맨시티가 많이 수행하는 것을 확인할 수 있다.
  - 이러한 분석을 통해, 코치틀은 상대팀이 어떤 패스를 할 때 위협적인지를 분석하고 이를 파악하여 전략을 세울 수 있다.
