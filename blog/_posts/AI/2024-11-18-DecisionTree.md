---
layout: post
title: Decision Tree
description: > 
  Decision Tree에 대해 정리하였습니다.
categories: [AI]
tags: [ML, Decision Tree]
---

'파이썬 머신러닝 완벽가이드(권철민 지음)', '창의적 문제 해결, 서영정 교수님, 2024-2',
[Decision Tree란 무엇인가요?](https://www.ibm.com/kr-ko/topics/decision-trees), [Decision Tree의 Impurity 지표](https://process-mining.tistory.com/106),
[ID3 알고리즘 설명](https://tyami.github.io/machine%20learning/decision-tree-2-ID3/)
{:.note title="출처"}

* this unordered seed list will be replaced by the toc
{:toc}

# Decision Tree
- 분류와 회귀 모두 사용 가능한 지도 학습 모델 중 하나입니다.
- 데이터에 있는 규칙을 학습을 통해 자동으로 찾아내 Tree 기반의 분류 규칙을 만듭니다.

![image](/assets/img/2024-11-18/Decision_Tree.png)

많은 규칙이 있다는 것은 복잡해진다는 것이고 과적합으로 이어지기 쉽습니다. Tree의 depth가 깊어질 수록 성능이 저하될 가능성이 높습니다.

가능한 한 적은 결정 노드로 높은 예측 정확도를 가지려면 데이터를 분류할 때 최대한 많은 데이터 세트가 해당 분류에 속할 수 있도록 결정 노드가 정해져야 합니다.
이를 위해 최대한 균일한 데이터 세트를 구성할 수 있도록 분할하는 것이 필요합니다.

결정 노드는 정보 균일도가 높은 데이터 세트를 먼저 선택할 수 있도록 규칙 조건을 만듭니다. 이러한 정보 균일도를 측정하는 대표적인 방법은 **엔트로피를 이용한 정보 이득(Information Gain) 지수**와 **지니 계수**가 있습니다.

## 분할 기준 알고리즘: 지니 계수, 정보이득
### 불순도(Impurity)
- 해당 범주 안에서 서로 다른 데이터가 얼마나 섞여 있는지를 의미합니다.
빨4 & 파1이면 불순도가 낮고, 순도가 높다라고 할 수 있으며
빨3 & 파5이면 불순도가 높고, 순도가 낮다고 할 수 있습니다.

**결정 트리는 불순도를 최소화(순도를 최대화)하는 방향으로 학습을 진행합니다.**

**불순도는 엔트로피, 엔트로피 등을 사용해서 측정**합니다.

### 불순도(Impurity) 지표
#### 엔트로피(Entropy)
불순도를 측정하는 방법 중 하나로, 정보의 불확실성을 측정합니다.

어떤 노드에서 분할 속성을 선택했을 때 정보량을 계산하고 해당 속성 값의 엔트로피를 최소화하는 방향으로 의사결정나무 구조를 만드는데 활용됩니다.

- 낮은 엔트로피 = 경우의 수가 적음 = 낮은 정보량 = 낮은 불확실성 = 낮은 불순도
- 높은 엔트로피 = 경우의 수가 많음 = 높은 정보량 = 높은 불확실성 = 높은 불순도

엔트로피를 측정하는 방법 중에 샤논 공식은 아래와 같습니다.

$$H(D) = -\sum_{k=1}^{m} p_k \log_2(p_k)$$

- D: 데이터 집합
- m: 분류하고자 하는 실제값들의 가짓수
- p_k: k번째 실제값에 해당하는 데이터들의 확률, 엔트로피를 계산하려는 데이터 집합에서 k번째 실제값에 해당하는 데이터의 비율로 계산

#### 속성별 엔트로피
- 속성 A로 분류했을 때 A의 모든 클래스의 각 엔트로피를 계산하고 데이터 개수만큼 가중치를 부여합니다.

$$h_A(D) = \sum_{values(A)} \frac{|D_v|}{|D|} * h(D_v)$$

- Values(A): 속성 A가 가질 수 있는 값의 집합
- D_v: 속성 A의 값이 v인 데이터의 집합
- \|D\|: 데이터 집합 D의 크기

#### 지니 계수
지니 계수는 불순도를 측정하는 또 다른 방법으로, 값이 0일수록 해당 노드는 완전히 순수하다고 볼 수 있습니다.

$$ Gini(D) = 1 - \sum_{i=1}^{C} P_k^2 $$

- C: 클래스의 총 개수
- p_i: 클래스 i에 속할 확률

### 정보이득(information Gain)
Impurity 지표들을 바탕으로 각 node들의 복잡성을 계산할 수 있습니다.

그리고 이 Impurity를 바탕으로, (decision tree에 의해 나누기 전의 Impurity - 나누어진 subset들의 Impurity) 값을 통해 Impurity가 얼마나 개선되었는지 계산할 수 있고, 이를 information gain이라고 합니다.

#### 속성별 정보 이득
- 정보이득이 클수록 속성 A를 기준으로 데이터를 분류했을 때 얻을 수 있는 정보량이 많습니다.
- A를 기준으로 나눌 때 엔트로피가 작으면 해당 속성을 기준으로 데이터를 나누기 좋습니다.

$$Gain(D, A) = h(D) - h_A(D)$$

## 유형
- ID3: **엔트로피와 정보 이득(획득)**을 메트릭으로 활용하여 후보 분할을 평가합니다.
- C4.5: 정보 획득 또는 획득 비율을 사용하여 분할 지점을 평가합니다.
- CART: '분류 및 회귀 트리'의 약자로 회귀도 가능한 알고리즘입니다. **지니 불순도**를 사용하여 분할할 이상적인 속성을 식별합니다. 지니 불순도를 사용하여 평가할 때 값이 낮을수록 이상적입니다.

### ID3(Iterative Dichotomiser 3)
- 성장(grow): 일반적으로 의사결정나무를 생성하는 방법을 '성장'이라고 부릅니다. 각 노드에서 엔트로피가 최대로 감소하도록 하는 분할 속성을 결정하는 과정입니다.

- **ID3**: 반복적으로 데이터를 나누는 알고리즘입니다. 탑다운 방식으로 정보이득을 최대화하는 greedy 방식의 최적화 알고리즘입니다. 각 노드마다 반복적으로 계산된 정보이득을 기반으로 노드별 데이터 분류 기준(속성)을 정합니다.

# Decision Tree 계산
## 와인 예제
![image](/assets/img/2024-11-18/와인예제1.jpg)
![image](/assets/img/2024-11-18/와인예제2.jpg)

## 컴퓨터 구매 여부 예제
<table>
  <thead>
    <tr>
      <th>번호</th>
      <th>나이(x1)</th>
      <th>수입(x2)</th>
      <th>학생 여부(x3)</th>
      <th>신용 등급(x4)</th>
      <th>구매 여부(y)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>청소년</td>
      <td>고소득층</td>
      <td>아니오</td>
      <td>좋음</td>
      <td>미구매</td>
    </tr>
    <tr>
      <td>2</td>
      <td>청소년</td>
      <td>고소득층</td>
      <td>아니오</td>
      <td>아주 좋음</td>
      <td>미구매</td>
    </tr>
    <tr>
      <td>3</td>
      <td>청년</td>
      <td>고소득층</td>
      <td>아니오</td>
      <td>좋음</td>
      <td>구매</td>
    </tr>
    <tr>
      <td>4</td>
      <td>중년</td>
      <td>중소득층</td>
      <td>아니오</td>
      <td>좋음</td>
      <td>구매</td>
    </tr>
    <tr>
      <td>5</td>
      <td>중년</td>
      <td>저소득층</td>
      <td>예</td>
      <td>좋음</td>
      <td>구매</td>
    </tr>
    <tr>
      <td>6</td>
      <td>중년</td>
      <td>저소득층</td>
      <td>예</td>
      <td>아주 좋음</td>
      <td>미구매</td>
    </tr>
    <tr>
      <td>7</td>
      <td>청년</td>
      <td>저소득층</td>
      <td>예</td>
      <td>아주 좋음</td>
      <td>구매</td>
    </tr>
    <tr>
      <td>8</td>
      <td>청소년</td>
      <td>중소득층</td>
      <td>아니오</td>
      <td>좋음</td>
      <td>미구매</td>
    </tr>
    <tr>
      <td>9</td>
      <td>청소년</td>
      <td>저소득층</td>
      <td>예</td>
      <td>좋음</td>
      <td>구매</td>
    </tr>
    <tr>
      <td>10</td>
      <td>중년</td>
      <td>중소득층</td>
      <td>예</td>
      <td>좋음</td>
      <td>구매</td>
    </tr>
    <tr>
      <td>11</td>
      <td>청소년</td>
      <td>중소득층</td>
      <td>예</td>
      <td>아주 좋음</td>
      <td>구매</td>
    </tr>
    <tr>
      <td>12</td>
      <td>청년</td>
      <td>중소득층</td>
      <td>아니오</td>
      <td>아주 좋음</td>
      <td>구매</td>
    </tr>
    <tr>
      <td>13</td>
      <td>청년</td>
      <td>고소득층</td>
      <td>예</td>
      <td>좋음</td>
      <td>구매</td>
    </tr>
    <tr>
      <td>14</td>
      <td>중년</td>
      <td>중소득층</td>
      <td>아니오</td>
      <td>아주 좋음</td>
      <td>미구매</td>
    </tr>
  </tbody>
</table>

![image](/assets/img/2024-11-18/컴퓨터예제1.jpg)
![image](/assets/img/2024-11-18/컴퓨터예제2.jpg)

# Decision Tree 실습
시각화는 Graphviz 패키지를 활용합니다.