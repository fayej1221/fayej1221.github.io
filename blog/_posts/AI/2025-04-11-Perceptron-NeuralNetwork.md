---
layout: post
title: 퍼셉트론 -> 신경망(활성화함수, 손실함수, 경사하강법)
description: > 
  Perceptron
categories: [AI]
tags: [Perceptron, activation function, loss function, minibatch, gradient descent]
---

[밑바닥부터 시작하는 딥러닝](https://www.yes24.com/Product/Goods/34970929)
BEYOND AI BASIC
{:.note title="출처 및 참고"}

* this unordered seed list will be replaced by the toc
{:toc}

# 퍼셉트론
- 다수의 신호를 입력으로 받아 하나의 신호를 출력
- 퍼셉트론을 이용하면 AND, NAND, OR의 3가지 논리 회로를 구현할 수 있음
- XOR의 경우 선형으로 나눌수가 없어서 퍼셉트론으로는 표현이 불가능

x1, x2라는 두 신호를 입력받아 y를 출력하는 퍼셉트론 수식으로 나타내면

$$
f(x) =
\begin{cases}
0 (b+w_1x_1+w_2x_2 <= 0) \\
1 (b+w_1x_1+w_2x_2 > 0)
\end{cases}
$$

- b는 편향: 뉴런이 얼마나 쉽게 활성화되느냐를 제어
- w_1, w_2는 각 신호의 가중치를 나타내며, 신호의 영향력 제어

- 가중치가 b일 때 입력이 1인 뉴런일 경우,
  - x_1, x_2, 1이라는 3개의 신호가 뉴런에 입력되어, 각 신호에 가중치를 곱한 후, 다음 뉴런에 전달됨.
  - 다음 뉴런에서는 이 신호의 값을 더하여, 그 합이 0을 넘으면 1을 출력하고, 그렇지 않으면 0을 출력
  - 더 간결하게 식을 표현하면

$$
y = h(b + w_1x_1 + w_2x_2)
$$
$$
h(x) =
\begin{cases}
0 (x <= 0) \\
1 (x > 0)
\end{cases}
$$

## 다층 퍼셉트론
- 그러므로 층을 더 쌓아서 XOR을 표현
- XOR은 AND, NAND, OR을 조합하여 표현이 가능

```python
def XOR(x1, x2):
  s1 = NAND(x1, x2)
  s2 = OR(x1, x2)
  y = AND(s1, s2)
  return y
```

# 신경망(Neural Network)
- 가장 왼쪽을 입력층
- 맨 오른쪽 줄을 출력층
- 중간 줄을 은닉층
- 신경망은 모두 3층으로 구성되는데 가중치를 갖는 층은 2개 뿐이기에 '2층 신경망'
- 그러나 문헌에 따라서 신경망 구성하는 층수를 기준으로 3층 신경망이라고 하는 경우도 존재

```python
# 가중치 초기화 함수
def init_parameters(num_hidden_units = 2):
  W1 = np.random.randn(2, num_hidden_units) # 첫번째 레이어 가중치
  B1 = np.zeros((num_hidden_units,1)) # 첫번째 레이어 바이어스
  W2 = np.random.randn(num_hidden_units, 1) # 두번째 레이어 가중치
  B2 = np.zeros((1, 1)) # 두번째 레이어의 바이어스
  return W1, B1, W2, B2 # 가중치 파라미터 리턴
```

## 활성화 함수
- 퍼셉트론에서 h(x) 함수처럼 입력 신호의 총합을 출력 신호로 변환하는 함수를 일반적으로 활성화 함수
- 퍼셉트론 함수의 식을 다시 써보면

$$a = b+w_1x_1+w_2x_2$$

$$y=h(a)$$

- 가중치가 달린 입력 신호와 편향의 총합을 계산하고, 이를 a라고 함
- a를 함수 h()에 넣어 y를 출력하는 흐름

# 활성화 함수
## 계단함수
**퍼셉트론**에서의 

$$
h(x) =
\begin{cases}
0 (x <= 0) \\
1 (x > 0)
\end{cases}
$$

를 **계단함수**라고 함

```python
def step_function(x):
  if x > 0:
    return 1
  else:
    return 0

def step_function(x):
  y = x > 0
  return y.astype(np.int)

```

## 시그모이드 함수
$$
h(x) = \frac{1}{1+exp(-x)}
$$

- e는 자연상수로 2.7182...의 값을 갖는 실수
- **신경망에서는 활성화 함수 자주 사용하는데 시그모이드 함수를 이용하여 신호를 변환하고, 그 변환된 신호를 다음 뉴런에 전달** 
- 그 외에 뉴런이 여러 층으로 이어지는 구조와 신호를 전달하는 방법은 기본적으로 앞에서 살펴본 퍼셉트론과 같음

```python
def sigmoid(x):
  return 1 / (1+np.exp(-x))
```

- 넘파이의 브로드캐스트: 넘파이 배열과 스칼라 값의 연산을 넘파이 배열의 원소 각각과 스칼라값의 연산으로 바꿔 수행하는 것

## 시그모이드와 계단함수
- '매끄러움'의 차이
- 둘 다 입력이 작을 때 0에 가깝고, 커지면 1에 가까워지는 구조
- 비선형 함수!

### 비선형 함수
- 신경망에서는 활성화 함수로 비선형 함수를 사용해야 함.
- 선형 함수를 이용하면 신경망의 층을 깊게하는 의미가 없음.
  -  층을 아무리 깊게 해도 은닉층이 없는 네트워크로도 똑같은 기능을 할 수 있다는 데 있음
    - h(x) = cx인 활성화 함수로 사용한 3층 네트워크
    - 식으로 나타내면 y(x) = h(h(h(x)))
    - 이 계산은 y(x) = c*c*c*x처럼 곱셈을 세 번 수행하지만 y(x) = ax와 같음(a=c^3)
    - 즉, 은닉층이 없는 네트워크로 표현할 수 있음

-> 선형 함수를 이용해서는 여러 층으로 구성하는 이점이 없으므로 층을 쌓는 혜택을 얻고 싶다면 활성화 함수로는 반드시 **비선형 함수**

## ReLU
- 입력이 0을 넘으면 그 입력을 그대로 출력
- 0 이하이면 0을 출력하는 함수

$$
h(x) = 
\begin{cases}
x (x > 0) \\
0 (x <= 0)
\end{cases}
$$

```python
def relu(x):
  return np.maximum(0, x)
```

# 다차원 배열의 계산
- 가로: 행, 세로: 열
## 행렬의 곱
- 행과 열을 원소별로 곱하고 값을 더해서 계산 -> np.dot()
- shape에 주의: 행렬 A의 열 수와 행렬 B의 행 수가 같아야 함

- 3x2 행렬 A와 2x4 행렬 B를 곱해 3x4의 행렬 C

# 3층 신경망 구현하기
- 입력 -> 출력 (forward Propagation; 순방향 처리)
- init_network(): 가중치와 편향을 초기화하고 딕셔너리 변수인 network에 저장, 이 딕셔너리 변수 network에는 각 층에 필요한 매개변수(가중치와 편향을) 저장
- forward(): 입력 신호를 출력으로 변환하는 처리 과정을 모두 구현

```python
def init_network():
  network = {}
  network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
  network['b1'] = np.array([0.1, 0.2, 0.3])
  network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
  network['b2'] = np.array([0.1, 0.2])
  network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
  network['b3'] = np.array([0.1, 0.2])

  return network

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def identity_function(x):
    return x

def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1 # affine transform
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2 # affine transform
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3 # affine transform
    y = identity_function(a3)

    return y

network = init_network()
x = np.array([1.0, 0.5])
y = forward(network, x)
print(y)   
```

- **affine transform**: 가중치 매트릭스인 W를 곱하고(선형변환) 여기에 바이어스 B를 더하는 연산, 변환결과는 원점이 이동된(바이어스 덧셈연산으로 인해) 선형변환!

# 출력층
- 회귀: 항등 함수
- 분류: 소프트맥스 함수

## 항등 함수와 소프트맥스 함수
**항등 함수**
- 입력과 출력이 항상 같다는 의미로 출력층에서 항등 함수를 사용시 입력 신호가 그대로 출력 신호가 됨
- 항등 함수의 변환은 은닉층에서의 활성화 함수와 마찬가지

**소프트맥스 함수**
$$
y_k = \frac{exp(a_k)}{\sum_{i=1}^{n} exp(a_i)}
$$

- exp(x)는 e^x를 뜻하는 지수 함수
- n은 출력층의 뉴런 수
- y_k는 그 중 k번째

```python
def softmax(a):
  exp_a = np.exp(a)
  sum_exp_a = np.sum(exp_a)
  y = exp_a / sum_exp_a
  return y
```

이렇게 코드를 작성하면 **오버플로우 문제**

- 소프트맥스 함수에 각각 C라는 임의의 정수를 곱하면
- C를 지수함수 exp()안으로 옮겨 logC로 만듦
- 마지막으로 logC를 C'라는 새로운 기호로 바꿈
- 이 의미는 소프트맥스의 지수함수를 계산할 때 어떤 정수를 더하거나 빼도 결과가 바뀌지 않는 다는 것
- C'에 어떤 값을 대입해도 상관없지만, 오버플로우를 막기 위해 입력 신호의 최댓값을 일반적으로 사용

```python
def softmax(a):
  c = np.max(a)
  exp_a = np.exp(a-c) # 오버플로 대체
  sum_exp_a = np.sum(exp_a)
  y = exp_a / sum_exp_a

  return y
```

### 소프트맥스 특징
- **0에서 1.0 사이의 실수, 총합은 1**.즉, 출력을 확률로 해석이 가능
- 지수 함수 y=exp(x)가 단조 증가 함수 이기에 소프트맥스 함수를 적용해도 각 원소의 **대소 관계는 변하지 않음**
- 신경망을 이용한 분류에서는 **일반적으로 가장 큰 출력을 내는 뉴런에 해당하는 클래스로만 인식**
- 소프트맥스 함수를 **적용해도 출력이 가장 큰 뉴런의 위치는 달라지지 않음**
- 결과적으로 신경망 분류시 출력층의 소프트맥스 함수를 생략해도 됨(현업에서는 지수 함수 계산에 드는 자원 낭비 때문에)
- **추론에서는 생략, 신경망 학습시 사용**

## 뉴론 수 정하기
- 분류에서는 분류하고 싶은 클래스만큼 설정하는 것이 일반적

## 순전파
- 학습 과정 생략, 추론 과정을 신경망의 순전파

# 신경망 학습
> 딥러닝을 종단간 기계학습(end-to-end machine learning)이라고 함(처음부터 끝까지 사람의 개입 없이 얻는다는 뜻)

## 손실 함수
- 손실함수를 기준으로 최적의 매개변수 값을 탐색

### 오차제곱합(Sum of Squares for error, SSE)
$$
E = \frac{1}{2}\sum_{k}(y_k - t_k)^2
$$

```python
def sum_squares_error(y, t):
  return 0.5 * np.sum((y-t)**2)
```

### 교차 엔트로피 오차(cross entropy error, CEE)
$$
E = - \sum_{k}{t_klogy_k}
$$
- t가 정답 레이블, 정답에 해당하는 인덱스의 원소만 1이고, 나머지는 0(원-핫 인코딩)
- y가 신경망 출력
- 실질적으로 정답일 때의 추정(t_k가 1일 때의 y_k)의 자연로그를 계산하는 식이 됨
- 예를 들어 정답 레이블이 2, 신경망 출력이 0.6이면 -log0.6 = 0.51이며 같은 조건에서 신경망 출력이 0.1이라면 -log0.1 = 2.30이 됨

- x가 1일 때 y는 0이 되고 x가 0에 가까워질 때 y의 값은 점점 작아짐

```python
def cross_entropy_error(y, t):
  delta = 1e-7
  return -np.sum(t*np.log(y+delta))
```
- delta를 더한 이유: np.log 함수에 0을 입력하면 -inf가 되어 계산이 불가능

## 미니배치 학습
- 데이터 일부를 추려 전체의 근사치로 이용
- 60,000장의 훈련 데이터 중에서 100개를 무작위로 뽑아 그 100장 만을 사용하여 학습하는 것

### 배치용 교차 엔트로피 오차
- 데이터가 하나인 경우와 데이터가 배치로 묶여 입력될 경우 모두를 처리할 수 있도록 구현

```python
def cross_entropy_error(y, t):
  if y.ndim == 1: # y가 1차원, 데이터 하나당 교차 엔트로피 오차를 구하는 경우
    t = t.reshape(1, t.size)
    y = y.reshape(1, y.size)
  
  batch_size = y.shape[0]
  return -np.sum(t*np.log(y+1e-7)) / batch_size # 정규화
```

- 원-핫이 아닌 숫자 레이블 인 경우, 
- 만약 원-핫이면 t가 0인 원소는 교차 엔트로피 오차도 0이므로 그 계산은 무시하는 것이 핵심
```python
def cross_entropy_error(y, t):
  if y.ndim == 1:
    t = t.reshape(1, t.size)
    y = y.reshape(1, y.size)
  
  batch_size = y.shape[0]
  return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
```
- np.log(y[np.arange(batch_size), t])
  - 0부터 batch_size-1까지 배열 생성 즉, batch_size가 5이면 np.arrange(batch_size)는 [0,1,2,3,4,5]를 생성
  - t에는 레이블 [2,7,0,9,4]
  - y[np.arange(batch_size), t]는 각 데이터의 정답 레이블에 해당하는 신경망의 출력을 추출하므로 [y[0,2],y[1,7]...]인 넘파이 배열을 생성

## 왜 손실함수를 설정?
> 신경망을 학습할 때 정확도를 지표로 삼아서는 안된다. 정확도를 지표로하면 매개변수의 미분이 대부분의 장소에서 0이 되기 때문이다.

왜 대부분의 장소에서 0?
- 100개중 32장을 제대로 인식하면 정확도는 32%
- 매개변수를 약간만 조정해서는 정확도가 개선되지 않고 일정하게 유지
- 정확도가 개선된다하더라도 그 값은 32.0123%와 같은 연속적인 변화보다는 33%나 34%처럼 **불연속적인 값으로 바뀜**

만약 손실함수를 지표로 삼으면
- 현재 0.92543...으로 나타남
- 매개변수가 조금 조정하면 그에 반응하여 손실 함수의 값도 0.93432처럼 **연속적으로 변화**

계단 함수와 동일
- **계단 함수를 활성화 함수로 사용하면 이전과 같은 이유로 신경망 학습이 잘 안됨.**
- 계단함수의 미분은 대부분의 장소에서 0
- 시그모이드는 어느장소라도 0이 되지 않음

## 수치미분
- 아주 작은 차분으로 미분

```python
# 나쁜 구현 예시시
def numerical_diff(f, x):
  h = 1e-50
  return (f(x+h)-f(x)) /h
```

**문제점**

- 1e-50, 반올림 오차 문제
  - 반올림 오차는 작은 값(소수점 8자리 이하)이 생략되어 최종 결과 계산에 오차가 생기게 함
  -> **10^-4** 사용

- f의 차분
  - x+h와 x 사이의 함수 f의 차분을 계산하고 있지만 오차가 있음
  - 해석적 미분은 x 위치의 함수의 기울기(접선)에 해당하지만
  - 이번 구현에서의 미분은 (x+h)와 x사이의 기울기에 해당
  - 진정한 미분(진정한 접선)과 이번 구현의 값은 엄밀히 일치하지 않음
  - **h를 무한히 0으로 좁히는 것이 불가능해 발생**

- **수치 차분에는 오차가 발생**
- (x+h)와 (x-h)일때의 함수 f의 차분을 계산하는 방법 사용
- **x를 중심으로 그 전후의 차분을 계산한다고 해서 중심차분**, 중앙차분이라고 함

```python
# 개선
def numerical_diff(f, x):
  h = 1e-4 # 0.0001
  return (f(x+h) - f(x-h) / 2*h)
```

> 수식을 전개해서 미분하는 것은 해석적 미분, 해석적 미분은 오차를 포함하지 않는 '진정한 미분'을 구함, 해석적 미분: 수학시간에 배운거, 수치 미분: 근사치로 계산

### 편미분
- 변수가 여러개인 함수에 대해 미분
- 어떤 변수를 미분할지 지정해야하고 지정한 변수를 제외한 나머지는 상수 취급하여 계산산
$$
f(x_0, x_1) = x_0^2 + x_1^2
$$

```python
def fucntion_2(x):
  return x[0]**2 + x[1]**2
  # or return np.sum(x**2)
```
## 기울기
- 앞에서는 x_0과 x_1을 따로 계산
- 동시에 하려면?
  - 모든 변수의 편미분을 베터로 정리한 것이 기울기!

```python
def numerical_gradient(f, x):
  h = 1e-4
  grad = np.zeros_like(x)

  for idx in range(x.size):
    tmp_val = x[idx]
    # f(x+h)
    x[idx] = tmp_val + h
    fxh1 = f(x)

    # f(x-h)
    x[idx] = tmp_val - h
    fxh2 = f(x)

    grad[idx] = (fxh1 - fxh2) / (2*h)
    x[idx] = tmp_val # 값 복원
  return grad
```
- 기울기는 가장 낮은 장소를 가리키지만 실제로 반드시 그렇다고는 할 수 없음
- 기울기는 각 지점에서 낮아지는 방향을 가리킴
- **기울기가 가리키는 쪽은 각 장소에서 함수의 출력 값을 가장 크게 줄이는 방향**

### 경사하강법
- 기계학습은 최적의 매개변수를 찾아냄
- 신경망 역시 최적의 매개변수(가중치와 편향)을 학습시 찾아야함
- 최적이란 손실함수가 최솟값을 떄의 매개변수 값
- 매개변수 공간이 광대하여 어디가 최솟값이 되는지 짐작이 불가능하여 **기울기를 잘 이용해 함수의 최솟값**을 찾으려는 것이 경사하강법

> 함수가 극솟값, 최솟값, 안장값이 되는 장소에서는 기울기가 0
- 안장점: 어느 방향에서 보면 극댓값이고 다른 방향에서보면 극솟값

- 기울어진 방향이 꼭 최솟값은 아니지만 가야만 한다.. 

### 경사법
- 이동한 곳에서 기울기를 구하고 또 나아가기를 반복하여 함수의 값을 점차 줄이는 것

$$
x_0 := x_0 - \eta \frac{\partial J}{\partial x_0}
$$
$$
x_1 := x_1 - \eta \frac{\partial J}{\partial x_1}
$$
- 에타는 갱신하는 양: 학습률로 한 번의 학습으로 얼마만큼 학습해야할지, 매개변수 값을 얼마나 갱신하냐를 정하는 것

```python
def gradient_descent(f, init_x, lr=0.01, step_num=100):
  x = init_x

  for i in range(step_num):
    grad = numerical_gradient(f, x)
    x -= lr * grad
  return x
```

### 신경망에서의 기울기
- 신경망에서도 기울기를 구해야함: 가중치 매개 변수에 대한 손실 함수의 기울기

형상이 2x3, 가중치 W, 손실함수가 L인 신경망

경사는 $$\frac{\eta L}{\eta W}$$로 나타냄

```python
import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient

class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2,3) # 정규분포로 초기화

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss

x = np.array([0.6, 0.9])
t = np.array([0, 0, 1])

net = simpleNet()

f = lambda w: net.loss(x, t)
dW = numerical_gradient(f, net.W)

print(dW)
```