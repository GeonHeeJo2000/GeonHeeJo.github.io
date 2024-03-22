---
layout: post
title: Diffusion
subtitle: Denoising Diffusion Probabilistic Models
---

이것은 Diffusion Model에 대한 설명입니다. 


### Generative Model
- 생성 모델(Generative Model)은 훈련 데이터의 분포를 따르는 유사한 데이터를 생성하는 모델이다.
- 생성 모델은 훈련 데이터과 같은 확률분포를 학습하므로써 새로운 sample을 만들어내는 문제이므로 데이터의 분포를 학습하는게 목적이다.

    <p align="center">
      <img src="https://image.slidesharecdn.com/pr409-221030173906-b096ada2/75/pr409-denoising-diffusion-probabilistic-models-8-2048.jpg?cb=1684179188">
      <br>
      Generative Model
    </p>
      
    - Generative Model사진은 생성모델의 대표적인 4가지를 설명한 그림이다.
      
        - GAN(Generative Adversarial Networks) : 생성자(Generative Model)는 noise(z)로부터 가짜이미지를 생성하면, 판별자(Descriminator)가 진짜와 가짜를 판단한다. 이렇게 두 네트워크는 적대적인 방식으로 서로를 개선하면서 학습한다.
        - VAE(Variational Auto-Encoder) : Encoder를 통해 입력데이터의 특성을 파악한 latent vector z를 만든 후에 z를 통해서 원본 데이터과 유사한 데이터를 생성한다. 
        - Flow-based models : VAE과 유사하지만, Encoder가 역함수(inverse)가 존재하는 function의 합성함수로 Encoder로 정의하고, Decoder에서 함성함수의 역함수로 정의한 방식이다.
        - Diffusion models : 데이터에 점진적으로 noise를 추가하고, noise만 남은 데이터를 바탕으로 다시 복원하면서 원본 데이터과 유사한 데이터를 생성한다

### Diffusion model
- Data에 noise를 조금씩 더하거나 noise로부터 복원해가는 과정을 통해 데이터를 생성하는 모델

    <p align="center">
      <img src="../assets/img/Diffusion image.JPG">
      <br>
      Diffusion Model
    </p>
      
    - Diffusion Model사진은 Diffusion Model의 아키텍처이다.
        - Diffusion모델은 두가지 단계를 통해서 진행되는데, 데이터의 noise를 추가하는 Forward process(diffusion process)과 noise만 존재하는 데이터로부터 noise를 제거하므로써 원본 데이터로 복구하는 Reverse process가 있다.


    **Forward Process(diffusion process)**

    <p align="center">
      <img src="https://img1.daumcdn.net/thumb/R1280x0/scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbBxn8h%2FbtrNwGmvbn9%2F43ZTjDwWXkrda4cQlhmpEK%2Fimg.png">
      <br>
      Forward Porcess
    </p>

    - Foward Process는 원본데이터($x_0$)으로부터 noise를 더해가면서 최종 noise($x_t$)로 가는 과정이다
    - $\beta_t$는 noise의 variance를 결정하는 파라미터로 얼만큼 noise를 더해가는지 결정한다. 즉, $\beta$가 1이면 한번에 noise가 된다는 의미이다.
    - 기존 Diffusion Model은 Forward Process에서 $\beta$를 학습하는것이 목적이다.
      
    ***$\beta$를 $10^-4$ ~ 0.02로 linear하게 증가시켜서 부여하는 방식으로도 사용되기도 한다.(학습을 하지 않고 고정된 상수값만 사용)***

    ```python
    def make_beta_schedule(schedule='linear', n_timesteps=1000, start=1e-4, end=0.02):
        if schedule == 'linear':
            betas = torch.linspace(start, end, n_timesteps)
        elif schedule == "quad":
            betas = torch.linspace(start ** 0.5, end ** 0.5, n_timesteps) ** 2
        elif schedule == "sigmoid":
            betas = torch.linspace(-6, 6, n_timesteps)
            betas = torch.sigmoid(betas) * (end - start) + start
        return betas
    
    def forward_process(x_start, n_steps, noise=None):
        x_sequence = [x_start] # initial 'x_seq' which is filled with original data at first.
        for n in range(n_steps):
            beta_t = noise[n]
            x_t_1 = x_sequence[-1]
            epsilon_t_1 = torch.rand_like(x_t_1)
    
            x_t = (torch.sqrt(1-beta_t) * x_t_1) + (torch.sqrt(beta_t) * epsilon_t_1)
            x_sequence.append(x_t)
        return x_sequence
    ```
    
    **Reverse Process(diffusion process)**
    
    <p align="center">
      <img src="https://img1.daumcdn.net/thumb/R1280x0/scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbBxn8h%2FbtrNwGmvbn9%2F43ZTjDwWXkrda4cQlhmpEK%2Fimg.png">
      <br>
      Forward Porcess
    </p>
    
    - Reverse Process는 noise($x_t$)만 있는 데이터에서 noise를 점점 제거하면서 원본 데이터로 복원하는 과정이다.
    - 기존 Diffusion Model은 가우시안분포를 학습하는 것이 목적이기 때문에 mean과 variance를 학습하는 것이 목적이다.
      
    *variance대신에 beta를 활용하기도 한다.(mean만 학습에 사용)*
    
    ```python
    def p_mean_variance(model, x, t):
        # Make model prediction
        out = model(x, t.to(device))
    
        # Extract the mean and variance
        mean, log_var = torch.split(out, 2, dim=-1)
        var = torch.exp(log_var)
        return mean, log_var
    ```
    
### VRNN
- RNN의 시간적 동적 특성과 VAE의 확률적 생성 모델링를 결합했다. 시간에 따라 변화하는 Trajectory를 효과적으로 학습하기 위해서 RNN도입
  
  ```markdown
  1. Prior : 데이터를 접근하기 전 가지고 있는 사전 분포를 통해서 데이터를 추정함.
      * Encoder가 입력데이터를 받아 Latent Space표현으로 변환하는 역할을 한다면, Prior은 Latent Space에 대한 전체적인 구조            와 분포를 정의함으로써 데이터를 생성할 때 일반화능력을 향상시킬 수 있다.
      * 수식은 t시점 이전(과거)의 정보만을 활용한 분포를 추정하는 식임을 확인할 수 있다.
  ```

  $$\ \text{p}_{\theta}(z_t | x_{<t}, z_{<t}) = \ \phi_{\text{prior}}(h_{t-1})$$

  ```markdown
  2. Latent Space : VAE과 같은 역할이다. 학습할 때는 Encoder의 확률 분포를 받고, 데이터 생성할 때는 Prior의 확률분포를 받는다.
  
  3. Encoder(Inference) : 입력 데이터를 내부 표현(잠재 공간)으로 변환
  * 학습할 때 사용
  ```
  
  $$\ \text{q}_{\theta}(z_t | x_{\leq t}, z_{<t}) = \ \phi_{\text{enc}}(x_t,h_{t-1})$$

  ```markdown
  4. Decoder(Generation) : Latend Space를 거친 z를 받아 원본 데이터과 같은 형태로 재구성
  ```
  
  $$\ \text{p}_{\theta}(z_t | z_{\leq t}, x_{<t}) = \ \phi_{\text{dec}}(z_t,h_{t-1})$$
  
  ```markdown
  5. Recurrence : 이전 시점의 hidden state과 입력데이터, Latent Vecor를 활용하여 현재 시점의 hidden state를 업데이터하는 과정이다
  ```
  
  $$\ \text{h}_t = f(x_t, z_t, h_{t-1})$$
  
- Loss : VAE의 손실 함수는 주로 두 부분으로 구성됩니다: 재구성 손실(reconstruction loss)과 정규화 손실(Kullback-Leibler divergence). 이 두 요소를 합쳐 Evidence Lower Bound (ELBO)라고 하며, VAE의 목표는 ELBO를 최대화하는 것입니다.
  
  ```markdown
  1. Reconstruction Loss
        - 입력 데이터와 출력 데이터 간의 차이를 줄입니다. 모델이 데이터를 얼마나 잘 재구성하는지 측정하는 지표
  
  2. Regularization Loss
        - 모델이 단순히 데이터를 재생산하는 것을 넘어서, 일반적인 데이터 패턴을 이해하고 새로운 데이터를 생성할 수 있게 하는 정규화 역할(입력 데이터의 복사본 생성 방지)
        - KL divergence는 Trajectory관점에서 다음 위치를 예측할 때, 이전 위치와 동일한 위치를 생성하지 않도록 하기 위한 것과 유사합니다. 즉, 모델이 데이터의 다양성을 유지하고 예측 가능한 패턴을 학습하도록 유도합니다.
  ```
  
- Model code link : [Model](https://github.com/emited/VariationalRecurrentNeuralNetwork/blob/master/model.py)
  

![image](https://github.com/GunHeeJoe/GunHeeJoe.github.io/assets/112679136/19c89399-9ba1-463e-8867-ea61078dec90)




### GVRNN
- GVRNN은 VRNN에 GNN기법을 추가한 것이다. 축구의 경우 각 선수들의 Trajectory는 다른 선수들의 영향을 받기 때문에 GNN기법도 추가한 것이다.
* 다중 에이전트의 상호작용을 학습
- Prior, Encoder(Inference), Decoder(Generation)를 통해 나온 확률분포에 GNN를 추가하므로써 선수들의 상호작용도 학습하는 구조
- Model code link : [Model](https://github.com/keisuke198619/C-OBSO/blob/main/vrnn/models/gvrnn.py)
  

![Model](https://github.com/GunHeeJoe/GunHeeJoe.github.io/assets/112679136/605202a0-3cf4-422e-87f5-fa1f8932cfcb)

