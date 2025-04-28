---
layout: post
title: DL| 2-stage detectors: R-CNN에서 Mask R-CNN까지의 발전
description: > 
  2-stage detectors: R-CNN, SPPNet, Fast R-CNN, Faster R-CNN, Mask R-CNN
categories: [DL]
tags: [R-CNN, SPPNet, Fast R-CNN, Faster R-CNN, Mask R-CNN, RoI Projection, RoI Pooling, RoI Align, RPN, NMS, Anchor box, SPP]
---
[RCNN을 알아보자](https://velog.io/@whiteamericano/R-CNN-%EC%9D%84-%EC%95%8C%EC%95%84%EB%B3%B4%EC%9E%90)
[SPPnet리뷰](https://deep-learning-study.tistory.com/445)
[Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition](https://velog.io/@lolo5329/%EB%85%BC%EB%AC%B8%EC%9A%94%EC%95%BD-Spatial-Pyramid-Pooling-in-Deep-Convolutional-Networks-for-Visual-Recognition)
[RoI Pooling](https://velog.io/@iissaacc/RoI-Pooling)
[NMS (Non-Maximum Suppression) & Anchor box](https://wikidocs.net/142645)
[Mask R-CNN 논문(Mask R-CNN) 리뷰](https://herbwood.tistory.com/20)
[Mask R-CNN 리뷰](https://ropiens.tistory.com/76)
[RoIPool과 RoIAlign 차이는?](https://velog.io/@oooops/RoIPool%EA%B3%BC-RoIAlign-%EC%B0%A8%EC%9D%B4%EB%8A%94)
BoostCampAITECH
{:.note title="출처 및 참고"}

* this unordered seed list will be replaced by the toc
{:toc}

# R-CNN(Regions with Convolutional Networks features)
> 설정한 Region을 CNN의 feature(입력값)로 활용하여 Object Detection을 수행하는 신경망

![image.png](/assets/img/2025/rcnn.png)

- Region Proposal: 물체 위치 찾기
- Region Classification: 물체 분류

![image.png](/assets/img/2025/rcnn2.png)
1. Region Proposal 방식 중에 Selective Search를 통해 RoI 추출
2. RoI의 크기를 조절해 모두 동일한 사이즈로 변형(CNN의 마지막인 FC layer의 입력 사이즈가 고정이므로 이 과정 수행)
3. RoI를 CNN에 넣어 feature를 추출
  - 각 region마다 4096-dim feature vector 추출(2000x4096)
  - Pretrained AlexNet 구조 활용(AlexNet 마지막에 FC layer 추가, 필요에 따라 파인튜닝 진행)
4. CNN을 통해 나온 feature를 SVM에 넣어 분류 진행
  - input: 2000x4096 features
  - output: Class(C+1) + Confidence scores
  - 클래스 개수(C개) + 배경 여부(1개)
5. CNN을 통해 나온 feature를 regression을 통해 bounding box 예측

- SVM: 분류나 회귀 분석을 위해 사용, 임의의 두 서포트벡터를 연결한 직선과, 그 직선을 그대로 평행이용하였을 때 처음 다른 벡터와 접하는 시점에서의 직선과의 거리가 최대가 될 때의 값을 구하는 기법

- CNN을 fine-turning 할 때 각 클래스별로 IoU가 0.5가 넘으면 positive sample로 두고, 그렇지 않으면 "background" 라고 labeled해두는데, 반면 SVM을 학습할 때는 IOU 기준이 0.3으로, ground-truth boxes(정답 박스)만 positive example로 두고, IoU가 0.3 미만이 영역은 모두 negative로 두었으며, 나머지는 전부 무시(이렇게 IOU 기준이 다른 것이 더 좋았다고 함...)
- bbox regressor는 IOU가 0.6이 넘으면 positive samples
- loss function: MSE Loss

**문제점**
1) 2000개의 Region을 각각 CNN 통과
2) 강제 Warping, 성능 하락 가능성
3) CNN, SVM classifier, bounding box regressor, 따로 학습
4) End-to-End X

**한계점**
1. Convolution Network의 입력 이미지가 고정 -> 고정된 이미지를 자르거나 조정해야함
2. RoI(Region of Interest)마다 CNN 통과 -> 오래걸림

# SPPNet
입력 크기에 관계없이 고정된 길이의 출력을 만들어낼 수 있는 방법으로 SPP를 적용한 네트워크를 SPPNet이라고 불림

![image.png](/assets/img/2025/sppnet2.png)
1. Selective Search를 사용하여 약 2000개의 region proposals를 생성
2. 이미지를 CNN에 통과시켜 feature map을 얻음
3. 각 region proposal로 경계가 제한된 feature map을 SPP layer에 전달
4. SPP layer를 적용하여 얻은 고정된 벡터 크기(representation)를 FC layer에 전달
5. SVM으로 카테고리를 분류
6. Bounding box regression으로 bounding box 크기를 조정하고 non-maximum suppression을 사용하여 최종 bounding box를 선별

## Spatial Pyramid Pooling

![image.png](/assets/img/2025/sppnet.png)

SPP는 Convolution layer에서 생성된 feature map을 입력받고, 각 feature map에 대해 pooling 연산을 하여 고정된 길이의 출력을 만들어 낼 수 있음

**작동 방식**
1. Convolution layer로부터 feature map을 입력 받음
2. 받은 입력을 사전적으로 정해진 영역을 나눔(4x4, 2x2, 1x1의 세가지 영역을 제공하고, 각각을 하나의 피라미드라고 부름)
3. 각 피라미드의 한칸을 bin, 각 bin에 대해서 max pooling 연산 적용
4. max pooling의 결과를 이어 붙여 출력

- 입력(feature map)의 크기가 k, bin의 개수를 M이라고 한다면 SPP의출력은 kM 차원의 벡터 -> 입력의 크기에 상관없이 사전에 설정한 bin의 개수과 채널값으로 SPP가 정해져서 항상 동일한 크기의 출력을 생성

**한계점**

3) CNN, SVM classifier, bounding box regression, 따로 학습
4) End-to-End X

# Fast R-CNN
![image.png](/assets/img/2025/fastrcnn.png)

1. 이미지를 CNN에 넣어 feature 추출(한번만), VGG16 사용
2. RoI Projection을 통해 feature map 상에서 RoI를 계산
3. RoI Pooling을 통해 일정한 크기의 feature가 추출(고정된 vector를 얻기 위한 과정, SPP 사용(pyramid level:1, target grid size:7x7))
4. Fully connected layer 이후, Softmax Classifier과 Bounding Box Regressor(클래스 개수: C+1개(클래스(C개)+배경))

- multi task loss 사용
  - classification loss + bounding box regression
- loss function
  - classification: cross entropy
  - BB regressor: Smooth L1
- Dataset 구성
  - IoU > 0.5: positive samples
  - 0.1 < IoU < 0.5: negative samples
  - Positive samples 25%, negative samples 75%
- Hierarchical sampling
  - R-CNN의 경우 이미지에 존재하는 RoI를 전부 저장해 사용
  - 한 배치에 서로 다른 이미지의 RoI가 포함됨
  - Fast R-CNN의 경우 한 배치에 한 이미지의 RoI만을 포함
  - 한 배치 안에서 연산과 메모리를 공유할 수 있음

**한계점**

4) End-to-End X

## RoI Projection
resnet을 backbone으로 쓴다고 했을 때 spatial dimension만 따지만 입력 이미지의 크기는 224x224, 출력 feature map은 7x7

**이때 feature map은 입력 이미지에 비해 32배 작은데, 이 점을 이용하여 입력 image의 RoI를 feature map에 대응하는 RoI로 바꾸는 것**

RoI를 [52, 106, 117, 206]이라면 feature map 상의 RoI는 [1.625, 3.3125, 3.65625, 6.4375]

pixel을 1보다 작은 수로 쪼갤수는 없으니 floating point를 그대로 쓸 수 없기에 반올림, 그러면 feature map 상의 RoI는 [2, 3, 4, 6]이라고 대략 짐작하는 셈

## RoI Pooling
**FC layer에 집어넣기 위해 max pooling을 함, feature map에서 가장 강한 신호를 하나 골라내는 작업으로 classficiation 성능이 좋음**

pooling layer의 output이 HxW, feature map위의 RoI가 hxw일 때 pool size를 계산하는 방법으로 h/H, w/W 이렇게 제안함

![image.png](/assets/img/2025/fastrcnn2.png)

## RoI Align 고안
feature map에 project한 RoI를 다시 입력 image로 옮겨보면 [64, 96, 128, 192]로 원래 RoI [52, 106, 117, 206]와는 조금 다름

단순히 bbox를 그리려면 조금 차이나는 것은 detection에서는 참고 넘어가도 segmentation은 **오차에서 차이가 나서 Mask R-CNN에서 정확한 segmentation을 위해서 RoI Projection을 할 때 가중치를 주는 것 같은 느낌의 RoI Align을 고안**

# Faster R-CNN
![image.png](/assets/img/2025/fasterrcnn.png)

1. 이미지를 CNN에 넣어 feature maps를 추출(CNN을 한번만 사용)
2. RPN을 통해 RoI 계산(기존의 selective search 대체, anchor box 개념 사용)

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

## Region Proposal Network (RPN)
![image.png](/assets/img/2025/fasterrcnn2.png)
1. CNN에서 나온 feature map을 input으로 받음(H: 세로, W: 가로, C: 채널)
2. 3x3 conv 수행하여 intermediate layer 생성
3. 1x1 conv 수행하여 binary classification 수행(2 (object or not) x 9 (num of anchors) 채널 생성, 4 (bounding box) x9 (num of anchors) 채널 생성)

- RPN 단계에서 classficiation과 regressor 학습을 위해 앵커 박스를 positive/negative samples로 구분
- 데이터 구성
  - IoU > 0.7 or highest IoU with GT: positive samples
  - IoU < 0.3: negative samples
  - Otherwise : 학습데이터로 사용 X

**loss**

![image.png](/assets/img/2025/fasterrcnn4.png)

## NMS(Non-Maximum Suppression)

**Faster RCNN에서의 과정**

- 유사한 RPN Proposals 제거하기 위해 사용
- Class score를 기준으로 proposals 분류
- IoU가 0.7 이상인 proposals 영역들은 중복된 영역으로 판단한 뒤 제거

### 예시
1. Confidence Score의 threshold를 0.4라면 0.4 이하인 bounding box를 모두 제거
2. Bouding box를 confidence score 기준 내림차수순으로 정렬
3. confidence score 0.9인 bounding box를 기준으로 잡고 뒤의 모든 박스를 비교
  - 0.8 박스와는 겹치지 않으므로 남겨둠
  - 0.7 박스와 IoU가 threshold 이상이므로 이 박스는 0.9 박스와 같은 것을 가리킨다고 간주하고 제거함
  - 0.65 박스, 0.6 박스(왼쪽)과는 겹치지 않으므로 남겨둠
  - 0.6 박스(오른쪽)와 IoU가 또 threshold 이상이므로 제거함
4. 0.8 박스를 기준으로 뒤의 모든 박스와 비교

### 문제점 -> anchor box로 해결
object끼리 겹칠 때 다른 object에 대한 bounding box까지 날라갈 수 있다는 것 -> anchor box

![image.png](/assets/img/2025/fasterrcnn3.png)
- anchor box가 2개라면 vector는 원래 벡터에서 두개를 이어붙인 것과 같이 생김
$$[P_c, b_x, b_y, b_w, b_z, c_1, c_2, c_3, p_c, b_x, b_y, b_w, b_z, c_1, c_2, c_3]$$
- detection을 할 때는 detection을 통해 예측한 object의 boundary box가 anchor box1에 유사한지 2에 유사한지 IoU 비교
- IoU 계산 후 높은 IoU를 갖는 anchor box 자리에 할당

## Training

Region proposal 이후 Fast RCNN 학습을 위해 positive/negative samples로 구분
- 데이터셋 구성
- IoU > 0.5: positive samples → 32개
- IoU < 0.5: negative samples → 96개
- 128개의 samples로 mini-batch 구성

Loss 함수
- Fast RCNN과 동일

- RPN과 Fast RCNN 학습을 위해 4 steps alternative training 활용
  - Imagenet pretrained backbone load + RPN 학습
  - Imagenet pretrained backbone load + RPN from step 1 + Fast RCNN 학습
  - Step 2 finetuned backbone load & freeze + RPN 학습
  - Step 2 finetuned backbone load & freeze + RPN from step 3 + Fast RCNN 학습

- 학습 과정이 매우 복잡해서, 최근에는 Approximate Joint Training 활용

**한계점**
1) 2000개의 RoI 각각 CNN 통과
2) 강제 Warping, 성능 손실 가능성
3) CNN, SVM classifier, bounding box regression, 따로 학습
4) End-to-End

다 해결됨!

# 정리
![image.png](/assets/img/2025/rcnn3.png)

# Mask R-CNN
- instance segmentation task 분야

## 구조
![image.png](/assets/img/2025/maskrcnn.png)

Faster R-CNN의 각 RoI에 대해 pixcel 단위 segmentation mask를 예측하는 branch 추가

mask branch는 classificaiton, bounding box regression breanch와 독립적이며 samll FCN(fully convolution network)

mask rcnn은 구현이 쉽고, mask branch는 small FCN이기에 연산 속도가 뛰어남

## RoIPool

![image.png](/assets/img/2025/roipool.png)

Faster R-CNN은 object detection을 위한 모델이였기 때문에 RoIPool과정에서 정확한 위치정보를 담는 것은 별로 중요하지않았음

- Quantization을 해주고 pooling하는데 만약 RoI가 소수점이면 각 좌표를 반올림
-> 왜곡이 발생

- classficaiton에는 문제가 없지만 픽셀별로 detection하는 경우 문제제

## RoIAlign
인스턴스 세그멘테이션의 대표적인 방식 Mask R-CNN에서는 RoIAlign방식을 사용

![image.png](/assets/img/2025/roialign.png)
![image.png](/assets/img/2025/roialign2.png)
![image.png](/assets/img/2025/roialign3.png)


1. RoI 영역을 pooling layer의 크기에 맞추어 등분
2. 각 그리드에 sample point를 잡음, 그림을 봤을 때 한 그리드에 4개의 샘플 포인트, 총 16개의 샘플포인트가 있음
3. sample point 하나를 기준으로 가까운 그리드 셀 4개에 대해서 bilinear interpolation(양선형 보간법) 계산을 해서 Feature Map을 계산

3가지 과정을 모든 sample point에 대해 진행, 하나의 영역 안에 4개의 값이 생기게 되는데 max 또는 average pooling을 사용해 2x2의 output을 얻어낼 수 있음

> Mask-R-CNN의 RoIAlign은 Quantization하지 않고도 RoI를 처리할 고정 사이즈의 Feature map을 생성할 수 있게 됨