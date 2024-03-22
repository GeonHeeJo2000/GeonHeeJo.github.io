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
    - $\beta$를 $10^-4$ ~ 0.02로 linear하게 증가시켜서 부여하는 방식으로도 사용되기도 한다.(학습을 하지 않고 고정된 상수값만 사용)

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
    - variance대신에 beta를 활용하기도 한다.(mean만 학습에 사용)
    
    ```python
    def p_mean_variance(model, x, t):
        # Make model prediction
        out = model(x, t.to(device))
    
        # Extract the mean and variance
        mean, log_var = torch.split(out, 2, dim=-1)
        var = torch.exp(log_var)
        return mean, log_var
    ```
    
### Loss
- Forward Process과 Reverse Process를 학습하기 최적화하기 위한 loss

    <p align="center">
      <img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbHCh8p%2FbtrNv8RlXMI%2FgSKxU6CFtxUMQPhOtWNUV0%2Fimg.png">
      <br>
      Loss
    </p>

    - $L_T$ : 원본데이터($x_0$)가 주어졌을 때 Forward Process에서 noise($x_t$)를 생성하는 분포과 Reverse Process에서 noise($x_t$)를 생성하는 분포간의 차이 -> 두 확률분포가 유사하도록 학습
    - $L_/(t-1)$ : Forward Process에서 구한 확률분포 q과 Reverse Process에서 구한 확률분포 p의 차이
    - $L_0$ : 원본데이터($x_0$)가 주어졌을 때 Forward Process에서 noise($x_t$)를 생성하는 분포과 Reverse Process에서 noise($x_t$)를 생성하는 분포간의 차이
