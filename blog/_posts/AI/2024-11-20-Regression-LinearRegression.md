---
layout: post
title: Regression, Linear Regression
description: > 
  회귀와 가장 기본적인 Linear Regression에 대한 설명 및 선형 회귀 모델의 종류에 대한 간략한 정리를 했습니다.
categories: [AI]
tags: [ML, Regression, Linear Regression]
---

'파이썬 머신러닝 완벽가이드(권철민 지음)', '창의적 문제 해결, 서영정 교수님, 2024-2', [머신러닝 선형 회귀](https://danawalab.github.io/machinelearning/2022/09/13/MachineLearning-LinearRegression.html), [cost function과 loss function 차이](https://uumini.tistory.com/51)
{:.note title="출처"}

* this unordered seed list will be replaced by the toc
{:toc}

# 회귀(Regression)
**여러 개의 독립변수와 한 개의 종속변수 간의 상관관계를 모델링하는 기법을 통칭**합니다.

$$Y = W_1*X_1 + W_2*X_2 + W_3*X_3 + ... + W_n*X_n$$에서 $$W_1, W_2, ... W_n$$은 독립 변수에 영향을 미치는 회귀 계수(Regreesion coefficients)입니다. 머신러닝 회귀 예측에서의 핵심은 피처와 결정 값 데이터 기반에서 학습을 통해 **최적의 회귀 계수**를 찾아내는 것입니다.

회귀는 회귀 계수의 선형/비선형 여부, 독립변수의 개수, 종속변수의 개수에 따라 여러가지 유형으로 나뉩니다. (선형->**선형 회귀**/비선형->**비선형 회귀**, 독립변수 1개->**단일 회귀**/ 독립변수 여러개->**다중 회귀**)

지도학습은 두 가지 유형으로 나뉘는데 바로 분류와 회귀, 두 기법의 큰 차이는 예측값이 카테고리와 같은 이산형 클래스 값이고 회귀는 연속형 숫자 값이라는 것입니다.

## 단순 선형 회귀 분석
$$y = wx + b$$

- 단순 선형 회귀 분석은 **독립 변수 x가 한 개**인 경우입니다.
- 독립변수 x와 곱해지는 값 w를 머신 러닝에서는 가중치(weight), 별도로 더해지는 값 b를 편향(bias)이라고 합니다.
- 직선의 방정식에서는 각각 직선의 기울기와 절편을 의미합니다.

## 다중 선형 회귀 분석
$$y = w_1x_1 + w_2x_2 + ... + w_nx_n + b$$

- 다중 선형 회귀 분석은 **독립 변수 x가 여러개**인 경우를 말합니다.

# 단순 선형 회귀를 통한 회귀 이해
$$\hat{Y} = w_0 + w_1 * X$$

위의 함수식의 **회귀 계수는 w_0: 절편(intercept), w_1: 기울기**입니다. 그리고 회귀 모델을 위와 같은 1차 함수로 모델링했다면 실제 주택 가격은 이러한 1차 함수 값에서 실제값만큼의 오류 값을 뺀(또는 더한) 값이 됩니다.(w_0 + w_1 * X + 오류 값).

**잔차**: 실제 값과 회귀 모델의 차이에 대한 오류 값입니다.

**최적의 회귀 모델**: 잔차 합이 최소가 되는 모델을 만든다는 의미이며 오류 값 합이 최소가 될 수 있는 최적의 회귀 계수를 찾는 다는 의미입니다.

### 손실(Loss) VS 비용(Cost)
- 손실 함수(Loss Function) : 개별적인 차이를 정의
- 비용 함수(Cost Function) : 손실함수를 제곱합 or 평균 등의 형식으로 정의
- 따라서 오류를 최소화한다고 했을 때 loss function을 최소화 한다는 말이 더 적합합니다.

## 비용 함수(Cost function, J(θ), 목적함수)
오류 값은 +나 -가 될 수 있습니다. 그래서 보통 오류 합을 계산할 때 절대값을 취해서 더하거나(MAE; Mean Absolute Error), 오류 값의 제곱을 구해서 더하는 식(RSS; Residual Sum of Square)을 취합니다.

일반적으로 미분 등의 계산을 편하게 하기 위해 RSS로 오류 합을 구합니다. 즉 $$Error^2 = RSS$$입니다.

![image](/assets/img/2024-11-20/RSS.png)

RSS는 회귀식의 독립변수 X, 종속변수 Y가 아닌 w 변수(회귀 계수)가 중심 변수입니다.

$$ RSS(w_0, w_1) = \frac{1}{N}\sum^N_{i=1} (y_i - (w_0 + w_1 * x_i))^2 $$

일반적으로 RSS는 학습 데이터 건수로 나눠서 위와 같이 정규화된 식으로 표현됩니다.

## 비용을 최소화 하기 - Gradient Descent
경사하강법을 통해서 하는 설명이 책에 기재되어있으나 경사하강법 이외의 방법과 함께 다른 포스트에 제대로 정리하고자 합니다.

# 선형회귀 종류
- **일반 선형 회귀**: 예측값과 실제 값의 **RSS(Residual Sum of Squares)를 최소화**할 수 있도록 회귀 계수를 최적화하며, **규제(Regularization)를 적용하지 않은** 모델입니다.
- **릿지(Ridge)**: 선형 회귀에 **L2 규제를 추가**한 회귀 모델입니다. L2 규제는 상대적으로 큰 회귀 계수 값의 예측 영향도를 감소시키기 위해서 회귀 계수값을 더 작게 만드는 규제 모델입니다.
- **라쏘(Lasso)**: **L1 규제를 추가**한 회귀 모델입니다. L1 규제는 예측 영향력이 작은 피처의 회귀 계수를 0으로 만들어 회귀 예측 시 피처가 선택되지 않게 하는 것입니다. 이러한 특성 때문에 L1 규제를 피처 선택 기능으로 부르기도 합니다.
- **엘라스틱넷(ElasticNet)**: **L1, L2 규제를 함께 결합**한 모델입니다. 주로 피처가 많은 데이터 세트에서 적용되며, L1 규제로 피처의 개수를 줄임과 동시에 L2 규제로 계수 값의 크기를 조정합니다.
- **로지스틱 회귀(Logistic Regression)**: **회귀라는 이름이 붙어있지만 분류에 사용**되는 선형 모델입니다. 매우 강력한 분류 알고리즘으로 일반적으로 이진 분류뿐만 아니라 희소 영역의 분류, 예를들어 텍스트 분류와 같은 영역에서 뛰어난 예측 성능을 보입니다.

자세한 내용은 다른 포스트에 작성하려합니다.

# Linear Regression 실습
```python
from sklearn.linear_model import Regression

'''
class sklearn.linear_model.LinearRegression(*, fit_intercept=True, copy_X=True, n_jobs=None, positive=False)
'''
```

LinearRegression 클래스는 RSS를 최소화하여 OLS(최소제곱법, Ordinary Least Squares) 추정 방식으로 구현한 클래스입니다.

## 속성
- coef_: 회귀 계수 값(추정 계수)
- intercept_: intercept(절편)의 값

## 실습 코드 일부분
```python
# 선형 회귀 OLS로 학습/예측/평가 수행
lr = LinearRegression()
lr.fit(X_train, y_train)
y_preds = lr.predict(X_test)
mse = mean_squared_error(y_test, y_preds)
rmse = np.sqrt(mse)
```

## 직접 함수 정의
```python
w_1 = 10 # 가중치 초기화
w_0 = 10
lr = 0.0025 # 학습률(일반적인 경우 0 < lr < 1)

def linear_regressor(X):
		return w_1 * X + w_0

def mean_squared_error_loss(y_pred, y):
		return (y_pred - y) * (y_pred - y)

def gradient_descent(X, y, y_pred):
		global w_1, w_0
		w_1 = w_1 - lr * 2 * X * (y_pred - y)
		w_0 = w_0 - lr * 2 * 1 * (y_pred - y)

# 모델 학습
for epoch in range(100):
		for X, y in zip(X_train, y_train):
				y_pred = linear_regressor(X) # 모델의 예측값 추출
				loss = mean_squared_error_loss(y_pred, y) # 손실값 계산
				gradient_descent(X, y, y_pred) # 손실값을 통한 경사하강법으로 w1, w0 업데이트

# 모델 성능 평가
y_pred = linear_regressor(X_test)
print(y_pred)
y_pred = linear_regressor(X_test).flatten()
print(y_pred)
result = mean_squared_error(y_test, y_pred)
print(result)
```