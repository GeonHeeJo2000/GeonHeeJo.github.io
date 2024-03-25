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

## Unlocking Downstream Analysis with Imputed Data
- 정확하게 imputation를 수행했는지 평가하고자한다.
- Downstream sporting analysis : xReceiver(Expected Recevier)
- xReceiver의 출력이 In-venue tracking-data의 출력과 일치하려면, imputed tracking-data는 trajectory space의 복잡한 특징을 생성해야한다.

  **xReceiver**
  
