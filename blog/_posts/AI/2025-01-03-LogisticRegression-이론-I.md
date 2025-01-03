---
layout: post
title: LogisticRegression 이론 I(Odds에서 Sigmoid(Logistic)으로)
description: > 
  LogisticRegression
categories: [AI]
tags: [LogisticRegression, Odds, Logit, Logistic, Sigmoid]
---

[정성적 분석과 정량적 분석 제대로 알고 하자](https://matthew530419.tistory.com/11),
[Odds, Logit(로짓), Sigmoid의 관계](https://m.blog.naver.com/je_un/222229440878),
[Logistic Regression, Logit, Odds, Logistic Function](https://chloe-ki.tistory.com/entry/odds-logit-logistic-function-linear-regression)
[4. Logistic Regression](https://supermemi.tistory.com/entry/AI-%EA%B8%B0%EC%B4%88-4-Linear-model-regression)
[Logistic Regression](https://incredible.ai/machine-learning/2016/04/26/Logistic-Regression/)
[로지스틱회귀분석](https://www.youtube.com/watch?v=xHWBOrX-hl8&t=344s&ab_channel=%EC%9D%B4%EA%B8%B0%ED%9B%88)
[Odds(오즈)란?](https://analysisbugs.tistory.com/167)
[로짓(Logit)이란?](https://haje01.github.io/2019/11/19/logit.html)
{:.note title="출처 및 참고"}

* this unordered seed list will be replaced by the toc
{:toc}

# Logistic Regression이란
> **logistic**을 사용하여 데이터가 어떤 범주에 속할 확률을 **0에서 1사이의 값으로 예측**하고, 그 확률에 따라 가능성이 **더 높은 범주에 속하는 것으로 분류해주는 지도 학습 알고리즘**

Linear Regression model(yhat=wx+b)는 y_hat을 quantitative(정량적)로 가정한다.
하지만 response variable이 qualitative(정성적 즉, categorical data(범주형))인 경우가 상당수 많다.

그래서 response variable이 정성적일 때에는 종속변수 y가 0 또는 1을 갖기에 단순 선형 함수인 y=wx+b로는 풀기가 힘들다. 입력값이 커지면 출력값 해석이 곤란해지기 때문이다. Logistic Regression을 이용하게 되면 0-1 사이의 값을 갖는 확률로 표현이 가능하다.

## 정성적(Qualitative) 분석과 정량적(Quantitative) 분석
- **정성적**
  - 특징: 데이터를 특징적으로 분석하는 기법, 비정형 데이터를 다루며 주관적인 의견이 반영될 수 있음, 탐색적 데이터 분석으로 불리기도 함
  - 예시: A/B/C/D 중 C에 해당되는 것 같다의 추정 결과를 확인하기 위한 분석법으로 이해할 수 있음
  - 장점: 시간과 비용이 적다.
  - 단점: 확실한 결과를 도출하기 위해서는 근거가 부족

- **정량적**
  - 특징: 확률 및 통계적 기법을 사용하여 분석하는 기법, 정형 데이터를 바탕으로 분석하기 때문에 객관적인 결과를 도출
  - 예시: A/B/C/D 중 모집단과 발생집단을 통계적 분석을 통해 분포도 및 확률을 구하고 C가 해당되는게 확실하다를 검증하기 위한 분석법으로 이해할 수 있음
  - 장점: 확실한 결과를 도출할 수 있음
  - 단점: 시간과 비용이 많이 듦

## 오즈(Odds)와 오즈비(Odds Ratio), 확률(Probability), 로짓(Logit)
- **Probability**: 원하는 횟수를 전체 횟수로 나눈 값(성공횟수/전체시도횟수)
- **Odds**: 실패비율 대비 성공 비율을 설명하는 것(승률의 개념으로 성공횟수/실패횟수)
- **Odds Ratio**: 독립 변수가 1단위 증가할 때 변화하는 Odds의 비율
- **Log-odds(Logit; logit + odds = logistic probit)**: Odds의 Log 값()

$$ Odds = \frac{p}{1-p} $$

$$ log odds = \ln(\frac{p}{1-p}) $$

**Odds, Odds Ratio**는 다르다.

### Range
- Probabillity: [0, 1]
- Odds: 좌변의 범위는 [0, inf], 우변은 [-inf, inf]
- Logit(log odds): [-inf, inf]

[-inf, inf]와 같은 형태가 되어 선형분석이 가능해진다.

## Logit, Logistic, Sigmoid
> Sigmoid Function?
"Sigmoid"는 "S형의"라는 형용사, "Functon"이라는 명사가 붙어 "x의 값을 변화시키면서 y값을 계산하여 그래프로 그리면 "S형"으로 그려지는 함수"이다. **대표적인 것이 Logistic이라 간단하게 Sigmoid라고 부른다.**

실패 대비 성공 확률은 Odds일 때 1보다 크고 작음에 따라 성공인지 실패인지 결정되고 log를 취하면(logit) 0이 성공과 실패의 기준이 된다.

exponential term(지수항)을 없애기 위해(선형화 하기 위해서) odds에 로그를 취했다.

이러한 **logit 함수와 sigmoid(logistic) 함수는 서로 역함수(y=x에 대하여 대칭) 관계**이다.

$$p = \frac{1}{1+e^{-L}} = \frac{e^{-L}}{e^{-L}+1}$$

logit 함수에서 sigmoid 함수를 유도하면 아래와 같다.

![image](/assets/img/2025-01-03/logit_to_sigmoid.png)

# Why used Sigmoid Function in Logistic Regression?
binary classification에서 p의 범위는 [0, 1]이다. 그래서 단순 y=xw+b로는 풀기가 어렵고, Odds, Logit을 이용한다.

- Probabillity: [0, 1]
- Odds: 좌변의 범위는 [0, inf], 우변은 [-inf, inf]
- Logit(log odds): [-inf, inf]

이런식으로 범위는 넓어진다. Logit을 사용하면 범위가 실수 전체가 된다. 단순 선형 함수로 풀기 힘든 binary classification 문제를 log(Odds(p)) = Wx+b로 선형회귀 분석할 수 있도록 만들어준다.

이 식을 위에서 유도한 것 처럼 p로 정리하면 아래의 수식(시그모이드 함수)가 나온다.

$$p(x) = \frac{1}{1+e^{-(Wx+b)}}$$

x 데이터가 주어졌을 때 성공확률을 예측하는 Logistic Regression은 Sigmoid 함수의 W와 b를 찾는 문제가 된다. 또한 시그모이드 함수는 [0, 1] 범위인 확률을 [-inf, inf]로 넓히기 때문에 보통 **멀티 클래스 분류 문제에서 softmax 함수의 입력으로 사용**된다.

# 시그모이드 함수의 활용
- 특정 사건이 발생할 확률을 표현, 0과 1사이의 값을 출력하기 때문에 정의역에 해당하는 확률 변수의 값에 따라 이산형 예측값의 확률을 표현하는데 활용(Logistic Regression)

- 인공신경망의 연산 과정에서 활성화 함수(activation function)로 활용, 시그모이드 함수 및 시그모이드 함수의 미분 형태가 모두 중요하게 사용됨