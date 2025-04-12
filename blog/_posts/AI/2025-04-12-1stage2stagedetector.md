---
layout: post
title: 1, 2 - Stage Detector(Regio Proposal, anchor box)
description: > 
  1, 2 - Stage Detector 차이
categories: [AI]
tags: [1 Stage Detector, 2 Stage Detector]
---
[Architecture - 1 or 2 stage detector 차이](https://velog.io/@qtly_u/Object-Detection-Architecture-1-or-2-stage-detector-%EC%B0%A8%EC%9D%B4)
[Selective Search (선택적 탐색)](https://developer-lionhong.tistory.com/31)
[시나브로_개발자 성장기:티스토리](https://developer-lionhong.tistory.com/35)
[C_3.01 Anchor Boxes](https://wikidocs.net/173914)
{:.note title="출처 및 참고"}

* this unordered seed list will be replaced by the toc
{:toc}

# 2 Stage Detector
**Regional Proposal과 Classification이 순차적으로 이루어짐**

대표적으로 R-CNN, Fast R-CNN, Faster R-CNN, R-FCN, Mask R-CNN등이 있음

이러한 과정 때문에 1-stage detector 방식에 비해서는 시간이 소요되지만, 보다 좋은 성능의 결과를 도출

## Region Proposal
**이미지 안에 객체가 있을 법한 영역(ROI)를 Bounding Box로 대략적으로 찾음**

### Selective Search
**영역의 질감, 색, 감도 등을 갖는 인접 픽셀를 찾아서 물체가 있을 법한 bbox나 segmentation을 찾아냄**

1. 원본 이미지로부터 각각의 object들이 1개의 개별 영역에 담길 수 있도록 수많은 영역을 생성(이때 object들을 놓치지 않기 위해서 over segmentation을 해줌)
2. 아래 그림의 알고리즘에 따라서 유사도가 높은 것들을 하나의 Segmentation(영역)으로 합침
  - 색상, 무늬, 크기, 형태를 고려하여 각 영역들 사이의 유사도를 계산
  - 유사도가 가장 높은 r_i와 r_j 영역을 합쳐서 새로운 r_t 영역을 생성
  - r_i와 r_j 영역과 관련된 유사도는 S 집합에서 삭제
  - 새로운 r_t 영역과 나머지 영역의 유사도를 계산하여 r_t의 유사도 집합 S_t 생성
  - 새로운 영역의 유사도 집합 S_t와 영역 r_t를 S, R 집합에 추가
3. 2번 과정을 여러번 반복하여 최종 후보 영역을 도출

- 이제 최종 후보 영역들에 대해서 CNN을 통핸 Classficiaton, Bounding Box Regression을 해주면 Object Detection이 수행
- 실제로 이것이 최초의 딥러닝 기반 Object Detection 알고리즘인 **R-CNN**의 전반적인 과정

### Sliding Window
**일정 크기의 window를 이동시키며, window 내에서 object를 detection하는 방식**

이때 window는 마치 CNN에서 커널이 입력 이미지와 연산하면서 움직이는 것처럼 왼쪽 상단에서부터 오른쪽 하단으로 이동

만약 window의 크기보다 객체의 크기가 훨씬 크다면 window에 들어온 객체의 일부분을 가지고는 객체를 잘 인식하지 못할 것

이를 위해서 **여러 형태의 window를 사용하거나, window 형태는 고정하되 입력 이미지의 크기를 변경하는 방식을 사용**

**계산량이 많아 시간이 오래 걸림**

# 1 Stage Detector
**Regional Proposal과 Classification이 CNN을 통해 동시에 이뤄짐**

**2-Stage보다 속도가 빠름**

Feature extraction에 해당하는 **Conv layers에서** 두가지가 동시에 이뤄짐

## Anchor box
ROI 영역을 검출하는 것이 아닌 **Anchor box 개념 사용**

앵커 박스는 **이미지의 각 위치에 미리 정의된 여러 크기와 비율의 바운딩 박스를 의미**

사전에 크기와 비율이 모두 결정되어 있는 박스를 전제로, 학습을 통해서 이 박스의 위치나 크기를 세부 조정하여 객체 조절

- 앵커는 k-means 클러스터링을 사용하여 COCO 데이터 세트에서 계산된 bounding box priors
- 클러스터 중심에서 오프셋으로 상자의 너비와 높이를 예측
- 필터 적용 위치에 대한 상자의 중심 좌표는 sigmod 함수를 사용하여 예측

### anchor box를 사용하는 방식
- 이미지 주위에 수천 개의 후보 앵커 박스 형성
- 각 앵커 상자에 대해 해당 상자에서 후보 상자의 약간의 오프셋 예측
- 실측 예를 기반으로 손실 함수 계산
- 주어진 오프셋 상자가 실제 개체와 겹칠 확률 계산
- 해당 확률이 0.5보다 크면 예측을 손실함수로 인수
- 예측된 상자에 보상 및 패널티를 부여하여 모델을 실제 객체만 현지화하도록 천천히 당김

### 위치는 어떻게?
네트워크 출력의 위치를 입력 이미지에 다시 매핑하여 결정됨

프로세스는 모든 네트워크 출력에 대해 복제됨. 결과는 전체 이미지에 걸쳐 타일링된 앵커 상자 세트를 생성

각 앵커 상자는 클래스의 특정 예측을 나타냄

예를 들어 이미지에서 위치당 두 개의 예측을 수행하는 두 개의 앵커 상자가 있으면 각 앵커 상자는 이미지 전체에 바둑판식으로 배열, 네트워크 출력의 수는 타일링된 앵커 박스의 수와 같음, 네트워크는 모든 출력에 대한 예측을 생성