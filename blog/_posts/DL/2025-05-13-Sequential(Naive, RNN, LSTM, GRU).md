---
layout: post
title: DL| 시퀀스 모델 총정리로, RNN부터 GRU(LSTM 구조 이해)
description: > 
    시퀀스 데이터를 처리하는 대표적인 모델들 간략하게 정리, LSTM의 내부 게이트 매커니즘 포함
categories: [DL]
tags: [Vision Transformer, ViT]
---
[Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
BoostCampAITECH
{:.note title="출처 및 참고"}

* this unordered seed list will be replaced by the toc
{:toc}

# Sequential Model

## Sequential Data

**연속적인 입력으로부터 연속적인 출력을 생성하는 모델로 일상생활에서 접하는 대부분의 데이터(Audio, Video, 동작 등)을 다루는 것을 목표**

얻고 싶은 것은 결국 하나의 라벨(one-out vector) 또는 하나의 정보인데, sequential data는 데이터가 언제 끝날지 알 수 없다. (input의 차원을 알 수 없음) → FCN, CNN을 사용할 수 없음

## 종류

### Naive sequence model

입력이 여러개일 때 다음 입력에 대한 예측을 하는 것으로 과거의 데이터를 고려해야하기 때문에 **input이 늘어날 수록 정보량이 늘어남**

![image.png](../../../assets/img/Sequential(Naive,%20RNN,%20LSTM,%20GRU)/image.png)

### Autoregressive model

현지 t에 대한 output은 **과거 n개만 의존한다고 가정하는 모델**

![image.png](../../../assets/img/Sequential(Naive,%20RNN,%20LSTM,%20GRU)/image%201.png)

### Markov model (first-order autoregressive model)

현재는 **바로 이전 과거에 대해서만 dependent한** Markov 성질을 가정한 모델

- 장점: 많은 정보를 고려하지 못하고 버려야 함
- 단점: joint distribution(결합확률분포; 두 개 이상의 확률변수에 대한 확률분포)를 표현하기가 쉬워짐

![image.png](../../../assets/img/Sequential(Naive,%20RNN,%20LSTM,%20GRU)/image%202.png)

### Latent autoregressive model

Markov model는 과거의 정보를 잘 활용하지 못함

해당 모델은 동일하게 과거 하나의 데이터만 영향을 받지만 **과거 데이터로 이전 정보를 요약한 hidden state를 사용**

![image.png](../../../assets/img/Sequential(Naive,%20RNN,%20LSTM,%20GRU)/image%203.png)

# Recurrent Neural Network(RNN)

Sequential Model을 구현할 수 있는 방법 중 하나가 바로 RNN, MLP와 거의 동일하지만 **자기 자신으로 돌아오는 구조가 있다는 차이**

![image.png](../../../assets/img/Sequential(Naive,%20RNN,%20LSTM,%20GRU)/image%204.png)

$h_t$는 $h_{t-1}$ 전달된 정보를 시간 순으로 늘어 놓았고, RNN 구조를 시간순으로 풀면 fully connected layer로 표현할 수 있음

## Short-term dependencies

과거 정보들을 다 취합하면 미래에서 사용해야하는데 **너무 길면 고려하기가 어려워짐**

![image.png](../../../assets/img/Sequential(Naive,%20RNN,%20LSTM,%20GRU)/image%205.png)

- 예를 들면 $h_1$은 출발점 $h_0$과 input $x_1$에 신경망의 weight를 곱하고 활성함수의 값을 사용해서 생성이 되는데 중첩되면 $h_t$에서 $h_0$의 영향은 활성함수와 weight를 t번 거친 값을 가지게 됨
- 이때 Activation Function으로 ReLU를 사용하면 양수 W가 계속 곱해지는데 W의 t제곱을 $h_0$에 곱하게 됨, 즉 $h_0$이 지수적으로 커져서 exploding gradient가 발생함
- 반면 Sigmoid를 사용하면 $h_0$은 매우 작아져서 의미가 없어지게 됨

![image.png](3a4785eb-8f43-46a1-a935-fff8e6cbb838.png)

## Long-term dependencies

그러한 문제를 해결한 것으로 멀리 떨어진 과거의 정보도 현재에 영향을 줄 수 있또록 설계됨

# Long Short Term Memory(LSTM)

## Vanilla RCNN 구조

이전의 Vanilla RNN의 구조는 다음과 같이 이전 call의 output 값이 현 시점의 input과 함께 연산되는 방법론인데 Long Term Dependency(장기 의존성)을 활용하여 sequence가 길어지면 앞의 정보가 뒤까지 충분히 전달되지 않는 모습을 해결 → **LSTM**

![image.png](../../../assets/img/Sequential(Naive,%20RNN,%20LSTM,%20GRU)/image%206.png)

## LSTM 구조

![image.png](../../../assets/img/Sequential(Naive,%20RNN,%20LSTM,%20GRU)/image%207.png)

![image.png](../../../assets/img/Sequential(Naive,%20RNN,%20LSTM,%20GRU)/image%208.png)

### Core idea

1. **Cell State**: 사진에서의 상단을 가로지르는  수평선, time step t까지의 정보를 요약함, output되지 않고 내부에서만 처리되어 다음 cell state로 전달됨, 일종의 컨베이어 벨트와 같음
    
    ![image.png](../../../assets/img/Sequential(Naive,%20RNN,%20LSTM,%20GRU)/image%209.png)
    
2. **Gate:** 정보를 선택적으로 통과시키는 방법으로 정보를 조작 추가 제거할 수 있음,  시그모이드 신경망과 pointwise multiplication operation으로 구성됨
sigmoid layer output은 0에서 1사이의 숫자를 출력하는데 각 구성 요소의 얼마나 많은 부분을 통과시켜야하는지를 나타냄(0은 아무것도 통과하지 않음, 1은 모든 것을 통과시킴을 의미)
    
    ![image.png](../../../assets/img/Sequential(Naive,%20RNN,%20LSTM,%20GRU)/image%2010.png)
    
    - **Forget Gate**
    - **Input Gate**
    - **Update Gate**
    - **Output Gate**

### Gate 설명

- **Forget Gate: Decide which information to throw away**
    - $h_{t-1}$: 이전의 output, $x_t$: 현재 입력을 이용해서 weight를 곱하고 activation을 통과시켜서 현재 cell state에 필요한 정보를 정함
    - activation이 현재 sigmoid로 0과 1 사이의 값(앞서 언급했던 것 처럼 0이면, 삭제 1이면 그대로 전달)
    
    ![image.png](../../../assets/img/Sequential(Naive,%20RNN,%20LSTM,%20GRU)/image%2011.png)
    
- **Input Gate: Decide which information to store in the cell state**
    - $i_t$: 또 다른 network를 통해서 어떤 정보를 추가할지
    - $\tilde{C}_t$: 현재 정보와 이전 출력값을 가지고 만드는 cell state의 예비 값
    - $i_t * \tilde{C}_t$: 각 상태 값을 얼마나 업데이트할지 결정한 비율에 따라 조정된 새로운 후보 값
    
    ![image.png](../../../assets/img/Sequential(Naive,%20RNN,%20LSTM,%20GRU)/image%2012.png)
    
- **Update Gate: Update the cell state**
    
    ![image.png](../../../assets/img/Sequential(Naive,%20RNN,%20LSTM,%20GRU)/image%2013.png)
    
- **Output Gate: Make output using the updated cell state**
    
    ![image.png](../../../assets/img/Sequential(Naive,%20RNN,%20LSTM,%20GRU)/image%2014.png)
    

# Gated Recurrent Unet(GRU)

- LSTM의 변형구조로 reset gate와 update gate만 존재하며 output gate가 없어서 LSTM에 비해서 파라미터의 개수가 적음
- hidden state가 곧 output이고, 과거를 summarize해서 다음 state에 전달되는 정보

![image.png](../../../assets/img/Sequential(Naive,%20RNN,%20LSTM,%20GRU)/image%2015.png)

# 결론

- RNN은 Sequential data를 다루기 위한 모델
- Vanilla RCNN은 Long-Term Dependency를 다루지 못한다는 단점이 있어 LSTM이 등장
- LSTM에서 파라미터를 줄이는 방향으로 일반화 성능을 높인 것이 GRU
- 많은 경우 LSTM보다 GRU를 사용할 때 성능 좋음
- 이후 Transformer 구조가 등장해서 적어도 언어 부분에서는 RNN이 거의 사용되지 않는 추세