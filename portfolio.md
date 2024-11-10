---
layout: page
title: 포트폴리오
description: > 
  안녕하세요, 진민주의 포트폴리오입니다.
---

---
<table>
  <tr>
    <td><img src="../assets/img/portfolio/프로필.jpg" alt="프로필" width="200"></td>
    <td>
      <h4>👋 Contact</h4>
      <ul>
        <li><b>💌 Email</b>: <a href="mailto:faye_1221@naver.com">faye_1221@naver.com</a><br></li>
        <li><b>🖊️ Tech Blog</b>: <a href="https://faye-1221.github.io/">faye-1221.github.io</a></li>
        <li><b>🧑‍💻 GitHub</b>: <a href="https://github.com/faye-1221">faye-1221</a></li>
      </ul>
      <h4>🙋🏻 How I work</h4>
      <ul>
        <li>항상 <b>더 나은 내가 되기 위해</b> 끊임없이 발전하고 노력합니다.</li>
        <li><b>긍정적인 영향</b>을 주는 일은 저에게 큰 동기부여가 됩니다.</li>
        <li><b>맡은 일에 항상 최선</b>을 다하며, <b>정해진 기한</b>을 반드시 지키려 합니다.</li>
      </ul>
    </td>
  </tr>
</table>

# 🔎 Profile
---
## 📖  교육 | Edcation
- **국립공주대학교** 2020.03 - 2025.02(졸업예정)
    - 주전공: 컴퓨터공학부 컴퓨터공학전공
    - 부전공: 응용수학과
- **부스트캠프 AI TECH 6기[네이버 커넥트 재단]** 2023.11 - 2024.04(수료)

  <img src="../assets/img/portfolio/부스트캠프_수료증.png" alt="부스트캠프_수료증" width="400">

---
## 📜 자격증 | License
- **정보처리기사 [2024. 09. 10, 한국산업인력공단]**

  <img src="../assets/img/portfolio/정보처리기사.png"   alt="정보처리기사" width="200">

- **빅데이터분석기사 필기 [2024. 09. 27, 한국데이터산업진흥원]**
    - 2024.11.30 실기 예정

---
## 🏆 수상 | Prize
- **우수 논문상(은상) [2022. 12. 02, 한국정보기술학회 대학생 논문경진대회]**
    - 논문명: ‘영상인식에서 전처리 유무 및 환경변화에 따른 객체 인식률 비교’

  <img src="../assets/img/portfolio/한국정보기술학회_우수논문상.png"   alt="한국정보기술학회_우수논문상" width="200">

---
## 🏙️ 경력 | Career
- **(주)에이리스, 동계 방학 인턴십 프로그램 [2023.01 - 2023.02]**
    - 위험 물품 X-ray 데이터셋 구축
    - 모델 평가를 위한 Metrics Python 코드 작성 및 Mask-RCNN fine-tuning
  
  <img src="../assets/img/portfolio/에이리스_수료증.png"   alt="에이리스_수료증" width="200">

---
# 👩🏻‍💻 프로젝트 | Projects
## [BoostCamp AI TECH 교육과정 최종 프로젝트] 충돌 예측 시스템

**GitHub**: [BoostCamp AI Tech finalproject GitHub](https://github.com/boostcampaitech6/level2-3-cv-finalproject-cv-12)

프로젝트 진행 기간: 2024.02. - 2024.03. (1개월)

<img src="../assets/img/portfolio/boost_final/거리측정.gif" alt="거리 측정 GIF" width="300">

<video width="320" height="240" controls>
  <source src="../assets/img/portfolio/boost_final/프로그램_데모.mp4">
</video>

_주의: 알림 소리가 발생하니 주의 바랍니다._  
_FPS: 30 내외_  
_Warning 상태 반경: 4m 이내 (노랑)_  
_Danger 상태 반경: 2m 이내 (빨강)_  
_Delay Time: 0.25s 내외_

### 프로젝트 개요

**프로젝트 소개**

- 단안 카메라로 실시간 충돌을 예측하는 알림 시스템입니다.

**배경**

- 자율 주행 기술이 적용된 배달 로봇 등의 상용화로 깊이 추정 기술에 대한 수요가 급증하고 있음
- 기존 거리 추정은 Lidar 센서를 이용하여 가능하나 비용이 비쌈
- 길거리에서 촬영된 영상은 사람들의 얼굴이 나오기 때문에 데이터로써 활용되기 어려움

→ 단안 카메라로 거리를 측정하고 실시간 충돌을 예측하며 얼굴 모자이크를 통해 개인 초상권 침해를 방지하는 시스템을 제작하였습니다.

**기대 효과**

- 단안 카메라 거리 측정 기술을 통한 안전 사고 방지
- 단안 카메라의 단점인 기상 악화 상황에서의 한계를 딥러닝을 이용해서 극복
- 비싼 Lidar 센서 대신 단안 카메라를 사용하여 비용 절감
- 영상속 사람들의 얼굴을 모자이크 처리하여 개인 초상권 침해 방지

**프로젝트 구조**

  <img src="../assets/img/portfolio/boost_final/프로젝트_구조.png"   alt="프로젝트_구조" width="300">


**프로그램 사용법**

1. back, front 레포지토리 README 참조하여 프로그램 실행
2. '영상 불러오기' 혹은 '카메라 불러오기'로 영상 선택
3. 왼쪽의 다양한 '기능설정' 선택 가능
4. '재생', '정지', '처음부터 다시 시작'을 통해 영상에 대한 조절 가능

    <img src="../assets/img/portfolio/boost_final/프로그램_사용법.png"   alt="프로그램_사용법" width="300">

### **역할 및 기여도**
총 5인 프로젝트로 모델 리서치 및 개발을 맡았습니다.

- Depth Anything 모델을 사용하여 실제거리로 근사    
  - 근사시킬 이미지를 촬영하여 실제 거리로 근사하였습니다.

    <img src="../assets/img/portfolio/boost_final/근사1.png"   alt="근사1" width="300">
    <img src="../assets/img/portfolio/boost_final/근사2.png"   alt="근사2" width="300">

### 성과 및 결과
Depth-Anything을 사용하기로 결정했을 때 상대거리인 문제점이 있었었습니다.

프로젝트에 적용하기 위해 Depth-Anything의 Metric Depth 모델을 사용해보거나 절대거리로 근사시키는 방법을 찾아 실험하였습니다.

### 배운점 및 느낀점
프로젝트에서 **Depth-Anything**의 **Metric Depth 모델**을 사용하면서, **상대거리 문제**를 해결하기 위해 절대거리로 근사시키는 방법을 실험했습니다. 최종적으로 절대거리로 근사시키는 방법을 사용했지만, 만족할 만큼 정확한 근사 값을 얻지는 못했습니다. 특히, **단안 카메라로 절대거리를 계산**하는 것이 어려운 문제임을 깨닫고, 이 분야에 대해 더 깊이 연구할 필요성을 느꼈습니다.

자율주행 분야에 대한 **흥미와 의지**가 낮았으나, 이번 프로젝트를 통해 흥미가 생겼고, **Depth Estimation**에 대한 관심도 커졌습니다. **Metric Depth 모델**은 특정 데이터셋에서는 잘 작동했지만, **직접 찍은 데이터**에는 적용되지 않았습니다. 만약 시간이 더 있었다면 **파인튜닝**을 시도해보았을 것 같다는 아쉬움이 남습니다. 또한, **ONNX**를 적용해본 경험이 있었고, 이를 통해 새로운 기술을 알게 되었습니다.


## [국립공주대학교 프로젝트] Farm_Security

**GitHub**: [Farm_Security](https://github.com/whdms2008/FarmSecurity)

프로젝트 진행 기간: 2024.02. - 2024.03. (1개월)

<img src="../assets/img/portfolio/farm_security/동물감지.png" alt="animal_detect_ex" width="300">
<br>
<video width="320" height="240" controls>
  <source src="../assets/img/portfolio/farm_security/프로그램_데모.mp4">
</video>

### 프로젝트 개요

**프로젝트 소개**

- **유해 동물로 인한 농작물 피해를 완화시키기 위한 AI 기반의 퇴치 시스템**입니다.

- YOLO를 이용하여 동물, 새, 사람으로 구분하고 동물과 새가 농경지 침입 여부를 확인합니다. 퇴치 단계별로 퇴치를 시행하며, 퇴치 대상 식별 및 퇴치 단계, 퇴치 여부를 실시간으로 어플리케이션을 통해 농경지 주인에게 알려줍니다.

1. 객체 탐지를 위해 AI 활용
   - 객체는 [사람/동물/새]로 구분
   - 객체 중 새, 동물은 퇴치 대상에 해당

2. 차영상 및 YOLO를 활용한 객체 판별 & 객체 탐지
   - 탐지 측면에서 효율성을 높이기 위해 차영상을 활용
   - 차영상을 통해 기존 배경과 차이 발생 시 YOLO를 실행하여 객체 판별 및 탐지 과정 진행

3. 퇴치 동작 다양화
   - 퇴치 객체 탐지 시 : 1-4단계 순서대로 퇴치 진행
     - 🚨 1단계 : 고강도 조명 출력
     - 🔉 2단계 : 랜덤 퇴치 신호 출력
     - ⚡ 3단계 : 고주파수 출력
     - 😈 4단계 : 1~3단계 종합 출력

4. 애플리케이션을 통한 퇴치 알림 서비스
- 퇴치 대상 식별 및 퇴치 단계, 퇴치 여부를 농장 주인에게 알림

**배경**

<img src="../assets/img/portfolio/farm_security/유해조수.jpeg" alt="유해조수" width="300">

_동물별 농작물 피해실태(환경부 생물다양성과/연도별 유해생물에 의한 피해현황(2014~2018)/2019)_

- 해당 그림에서 알 수 있듯이 해마다 유해조수로 인한 피해 규모가 증가 추세임을 알 수 있습니다.
- 이러한 피해 규모 완화를 위해서 사람이 직접 포획할 경우 인건비, 안전 등 고려해야할 요소가 많기에 무인 시스템의 필요성을 느껴 해당 프로젝트를 개발하게 되었습니다.

**기대 효과**

- 단안 카메라 거리 측정 기술을 통한 안전 사고 방지
- 단안 카메라의 단점인 기상 악화 상황에서의 한계를 딥러닝을 이용해서 극복
- 비싼 Lidar 센서 대신 단안 카메라를 사용하여 비용 절감
- 영상속 사람들의 얼굴을 모자이크 처리하여 개인 초상권 침해 방지

**프로그램 구조**

<img src="../assets/img/portfolio/farm_security/프로그램_구조.png"   alt="프로그램_구조" width="300">

  - 라즈베리(카메라)는 실시간으로 영상 촬영 및 AI 모듈에 영상 제공
  - AI 모듈은 전달받은 영상에서 먼저 차영상을 구함. 탐지된 차영상 있을 시 YOLO로 객체 판별 진행
  - 판별 객체가 퇴치 객체(동물 또는 새)일 경우, AI 모듈은 퇴치 단계에 따라 라즈베리(빛) 또는 라즈베리(스피커) 제어
  - 라즈베리(빛) 또는 라즈베리(스피커)는 AI 모듈의 제어 신호에 따라 작동됨
  - AI 모듈은 동물 또는 새 탐지 시 [카메라 일련번호 / 탐지 객체 캡처 링크 / 퇴치 단계 / 탐지 시간] 정보를 서버(=Spring Boot)에 송신
  - 서버는 실시간으로 AI 모듈에서 보내는 정보를 감지. 감지된 정보가 있을 경우 해당 정보를 DB에 삽입. 또한 이 정보를 사용자에게 파이어베이스 알림을 통해 전송 {-> 탐지 객체 있을 경우 사용자가 알아야 하므로}
  - 사용자가 과거 기록 확인 요청할 경우 해당 요청 정보 확인 가능

### **역할 및 기여도**
총 5인 프로젝트로 AI팀의 Main PA, Sub PL과 서기(회의 정리 & github 이슈 관리) 역할을 맡았습니다.

- **유해조수 데이터셋**을 만들고 **Yolov4로 학습을 진행**했습니다.
- 정확도 향상을 위해서 **전처리와 증강을 진행**했습니다.
- **환경변화에 따라 객체 인식률이 변화하는지** 실험했습니다.


#### 데이터셋 구축
1. **클래스 분류**  
   클래스는 animal, bird, human으로 나누어 진행했습니다.

   - bird와 animal 데이터는 Selenium을 활용하여 크롤링하였습니다.
     1. _한국 유해조수 자료를 참고하여 데이터셋을 수집했습니다._
     2. Animal: 멧돼지, 고라니, 청설모
     3. Bird: 까마귀, 까치, 직박구리, 어치, 참새
   - Human 데이터는 Kaggle 자료를 사용했습니다.  
     [Human Detection Dataset - Kaggle](https://www.kaggle.com/datasets/constantinwerner/human-detection-dataset)

2. **라벨링 작업**  
  labelimg 툴을 이용하여 라벨링을 진행했습니다.

#### yolov4 학습
1. yolov4 선택 이유
    
    모델을 yolo로 고정해야했기에 yolo에서도 어떤 모델을 선택할지 결정해야했습니다. 당시 yolov5까지 나와 v4와 v5를 비교했습니다. darknet 프레임워크를 기반으로 구현되어 있는 v4를 사용할 경우 FPS 성능 최적화가 가능하여 성능 분석 및 최적화 연구에 용이하다고 생각되었기에 v4로 선택했습니다.
    
    [YOLOv4와 YOLOv5 비교](https://yong0810.tistory.com/30)
    
2. yolov4 세팅
    
    [YOLOv4 학습 과정](https://velog.io/@jhlee508/Object-Detection-YOLOv4-Darknet-학습하여-Custom-데이터-인식-모델-만들기-feat.-AlexeyABdarknet)  
    해당 블로그를 참고하여 colab에서 학습을 진행했습니다. pre-trained 가중치 파일은 yolov4.conv.137(ImageNet 데이터셋으로 사전 학습된 CSPDarknet53 백본을 포함)를 사용했습니다.
    
3. yolo의 한계로 인해 더욱 향상하지 못한 점
    
    animal과 human은 AP가 0.8 정도로 높게 나왔으나 bird는 작은 객체이고 여러 마리가 모여있다보니 성능이 좋게 나오지는 않았습니다.
    
    **[yolo 한계]**
    
    YOLO는 영상을 7x7 의 그리드셀로 분할하여 각 그리드 셀을 중심으로 하는 각종 크기의 오브젝트에 대해서 경계박스 후보를 2개 예측한다.
    
    R-CNN 계열은 후보를 1천개 이상 제안하는것에 비해 YOLO는 총 7x7x2 = 98개의 후보를 제안하므로 이로 인해 성능이 떨어진다.
    
    그래서 한 오브젝트 주변에 여러개의 오브젝트가 있을 때 검출을 잘 못한다.
    
    예를 들면 새떼처럼 조그만 오브젝트들이 모여 있는 경우이다.
    
    영상에서 작게 나타난 새 크기에 비해 그리드 셀은 상대적으로 너무 크다.
    
    같은 이유로 셀 하나 안에 오브젝트가 여러개 있으면 최대 2개까지밖에 예측을 못한다
    
    - [출처] [YOLO, Object Detection Network](https://blog.naver.com/sogangori/220993971883)

    
#### 전처리 & 증강 rotate 실험
1. 데이터셋 리사이즈, 클래스 비율 맞추기, bilateralFilter를 통한 노이즈 제거, CFG 파일 재설정 (테스트 시 이미지 사이즈 늘리기)을 진행했습니다.

   - **결과**: 71% → 76% 성능 향상

2. 이미지 Rotate를 추가 후 Re-labeling을 진행했습니다.

   - **결과**: 76% → 88.2% 성능 향상

  → 경계를 강조하면 더 성능이 향상될 것으로 예상하여 샤프닝 및 소벨 필터를 적용했으나 성능이 향상되지는 않았습니다.
    
#### 환경변화에 따른 인식률 실험

눈, 비, 일출/일몰, 밤 필터를 데이터셋에 적용시켰습니다.

<img src="../assets/img/portfolio/farm_security/환경변화.png"   alt="환경변화" width="300">

mAP를 비교했을 때 필터를 추가했을 때 안 했을 때보다 3퍼 낮아지긴 했으나 큰 영향을 끼칠 확률은 저조하다고 결론내렸습니다.

### 성과 및 결과
- 데이터셋 이미지 사이즈를 동일하게 하고 가우시안 필터 방식의 양방향 필터를 사용하여 이미지의 엣지를 보존하고 선명도를 개선하였습니다. 추가적으로 train, test, valid 비율을 맞춰서 **약 6%의 성능을 향상**시켰습니다.
- **공주대학교 산학협력단 저작권 등록**되었습니다.
- 한국정보기술학회 대학생 논문경진대회 **우수 논문상(은상):** 논문명: ‘영상인식에서 전처리 유무 및 환경변화에 따른 객체 인식률 비교’ **– 주저자**
- 한국정보기술학회 대학생 논문경진대회 **우수 논문상(동상)**:논문명: ‘AI를 이용한 농작물 피해 완화 시스템’

### 배운점 및 느낀점
- 해당 프로젝트는 인공지능 수업을 듣고 프로젝트를 통해 더 공부하고 싶어서 진행하게 되었습니다. 컴퓨터 비전 분야에 대해서 알게 되었고 흥미를 돋우게 된 프로젝트입니다.
- 컴퓨터 비전에 대해서 아예 모르는 상태로 시작한 프로젝트라 많이 헤맸습니다. 가장 기본적인 데이터 사이즈를 동일하게 한다거나 클래스의 비율이 맞춰야 한다는 것을 몰랐습니다. 그래도 이미 증명된 것들을 실험을 통해서 직접 확인해본 것은 좋았던 경험이라고 생각합니다.
- 교수님께서 YOLO 모델로 고정해서 진행하라고 하셨다보니 다양한 모델을 사용해보지 못한 점이 아쉽습니다. 하지만 그덕에 기초적인 실험을 해볼 수 있었던 것 같습니다.
- 최종적으로 애매하게 마무리된 감이 있지만 개인적으로는 학습적으로 얻어간 것이 많았다고 생각합니다.

## [ 국립공주대학교 웹프로그래밍 기말 팀 프로젝트] Live-rary

프로젝트 진행 기간: 2022.12.

<img src="../assets/img/portfolio/live-rary/프로그램_배너.png" alt="프로그램_배너" width="300">

### 프로젝트 개요
- 사용자가 도서관의 소장 도서 목록을 인터넷에서 쉽게 확인하고, 원하는 책이 대출 가능한지 여부를 알 수 있는 프로그램을 개발하였습니다.

### **역할 및 기여도**
총 3인 프로젝트였습니다. 각자 맡은 역할은 아래와 같습니다.

- 김규빈: 초기 레이아웃 구성 및 CSS 최종 수정, 회원 정보 관리, 책 검색, 오류 수정
- 진민주: 유저 화면, 대출 목록, 예약 목록, 책 검색
- 홍상혁: 시나리오 작성, 사서 화면

### 성과 및 결과
- **공주대학교 산학협력단 저작권 등록**되었습니다.
- 프로그램 화면(발표 PPT)

<iframe src="../assets/img/portfolio/live-rary/발표.pdf" width="600px" height="400px"></iframe>
    
- 보고서

<iframe src="../assets/img/portfolio/live-rary/보고서.pdf" width="600px" height="400px"></iframe>

### 배운점 및 느낀점
- 해당 프로젝트를 진행할 때 기말고사 기간을 고려하여 4일 동안 다 구현하기로 했습니다. 진행하면서 이 기간안에 다 구현할 수 있을까? 걱정되기도 했습니다. 팀원 모두가 같은 생각이었는지 밤까지 계속 구현하고 있는 모습이 아직도 생생합니다.
- 그리고 어떻게 구현해야할지 막막했던 부분은 다른 팀원이 구현해놓은 코드를 참고하면서 했었습니다. 이를 통해서 jsp와 javascript를 사용하는 방식을 많이 알게되었습니다.
- 최종적으로 구현하고자 했던 것을 잘 마무리했고 저작권 등록과 학점이 좋게 나온 면에서 결과 또한 좋게 나왔습니다. 해당 프로젝트는 결과도 좋았지만 원만한 소통과 모두가 노력했다는 면에서 제일 퍼펙트한 프로젝트가 아니었나싶습니다.

---
# Competition
## [BoostCamp AI TECH 교육과정 비공개대회] Hand Bone Image Segmentation 대회

**GitHub**: [BoostCamp AI Tech semanticsegmentation](https://github.com/boostcampaitech6/level2-cv-semanticsegmentation-cv-12)

대회 순위: public: 2nd, private: 2nd

프로젝트 진행 기간: 2024.02.07 - 2024.02.22

<img src="../assets/img/portfolio/hand_bone/대회_배너.png" alt="대회_배너" width="300">

### 대회 개요
- 정확한 Bone Segmentation은 의료 진단 및 치료 계획을 개발하는데 필수적입니다. 대회에서 주어지는 데이터인 손 X-ray 이미지를 통해 사람의 나이와 키를 예측할 수 있습니다. 이를 위해서 정확히 Segmentation하는 모델을 만드는 것이 대회의 목표였습니다.
- 해당 대회에서 사용된 이미지는 보안상의 이유로 공개될 수 없습니다.

#### 개발 환경
- 컴퓨팅 환경 : 개인별 V100 서버 할당, VS code의 SSH Extension 사용
- 협업 툴 : Github, Notion, Slack, Zoom, WandB

#### 데이터셋
- Train : 800 Labeled Images
- Test : 288 Unlabeled Images
- 2048 x 2048, 3channel
- 개인별 왼손, 오른손 두 장의 이미지 set
- 29개의 클래스
```text
'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
'Triquetrum', 'Pisiform', 'Radius', 'Ulna'
```

#### 평가지표
- Test set의 Dice coefficient로 평가
    - Semantic Segmentation에서 사용되는 대표적인 성능 측정 방법
    - Dice

      <img src="../assets/img/portfolio/hand_bone/metric.png" alt="metric" width="300">

### 역할 및 기여도
팀은 총 5인으로 구성되었습니다.

- 성능 향상을 위한 **전처리** 및 **Data Augmentation** 실험을 했습니다.
- **Unet++**, **Multi Loss** 실험을 했습니다.

**데이터 분석**
- 데이터셋에서 성능이 저하될 수 있는 악세서리를 확인했습니다.
- 왼손, 오른손의 이미지를 각각 겹쳐서 손가락의 위치가 한 곳에 모여있지 않은 것을 확인했습니다.
    - 같은 방법으로 손등에 겹쳐져 있는 중복 클래스인 Pisiform, Trapzoid의 성능을 향상시키기 위해서 손등의 위치를 파악했습니다.

**전처리와 증강 실험**
1. **Resize**
    
    원본 이미지가 2048x2048인데 베이스 코드에는 512x512로 resize되어 있었습니다. 많은 정보가 손실될 수 있기 때문에 컴퓨터 환경에서 학습할 수 있는 최대 이미지 크기 1024x1024로 변경하여 학습하였습니다.
    
    → 0.0145% 향상
    
2. **Sharpen**
    
    뚜렷해야할 부분을 강조하면 성능이 향상될 것이라 생각했습니다. 이미지 크기 512에서는 향상되었지만, 1024에서는 향상되지 않았습니다.
    
3. **CLAHE**
    
    local contrast를 조정하고 AHE와 다르게 nosie amplification이 해결되는 CLAHE를 적용시켜서 소프트 조직과 뼈 사이의 대비를 적절하게 조절하고자 했습니다.
    
    <table>
      <thead>
        <tr>
          <th>Image Size</th>
          <th>Average Dice</th>
          <th>Method</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>512x512</td>
          <td>0.9507 → 0.9519</td>
          <td>CLAHE(gray scale로 변경해서 적용)</td>
        </tr>
        <tr>
          <td>1024x1024</td>
          <td>0.9652 → 0.9671</td>
          <td>CLAHE(gray scale로 변경해서 적용)</td>
        </tr>
        <tr>
          <td>1024x1024</td>
          <td>0.9652 → 0.9668</td>
          <td>CLAHE(LAB에서 L채널에만 적용)</td>
        </tr>
      </tbody>
    </table>
    
    CLAHE를 사용했을 때 성능이 향상된 모습을 보였습니다.
    
    또한 CLAHE를 사용하기 전에 전체적인 밝기를 낮춰서 적용시켰을 때 0.9671에서 0.9672로 미미하게 성능이 향상했습니다.
    
4. **HorizontalFlip**
    
    EDA를 통해 확인했을 때 왼손과 오른손의 차이가 많이 나지 않는다고 판단이 되어서 적용했습니다. → 0.0002% 향상
    
    HorizontalFlip을 적용시킨 데이터셋을 아예 추가해서 학습시켰습니다. → 0.002% 향상
    
5. **Rotate**
    
    EDA를 통해 확인했을 때 손가락의 경우 한 곳에 모여있지 않은 것을 확인했습니다. 다양한 위치에 있는 손가락을 균형있게 학습시키기 위해 적용했습니다. → 0.001% 향상
    
6. **Crop**
    
    손등에 겹쳐져 있는 중복 클래스인 Pisiform, Trapzoid의 성능을 향상시키는 것이 성능 향상에 매우 중요할 것이라고 판단했습니다. 
    
    EDA를 통해 손등의 위치를 파악하고 손등 부분을 Crop하거나 손등 이외의 부분을 마스킹해서 학습시켰습니다. 그러나 가설과는 다르게 오히려 성능이 하락하였습니다. segmentation 특성을 파악하고 접근하지 못했던 것 같습니다.
    
7. **Flip 한 쪽 방향으로 실험**
    
    Flip을 통해 양쪽 손의 데이터를 한쪽 손으로 통일시켜 학습 및 추론을 진행하였습니다. → 성능은 비슷하거나 저하되었습니다.

**모델 실험**

Unet 모델을 사용했을 때 성능이 좋다고 생각되어 이외의 방법들로 성능을 높이고자 했습니다.

**Combine Loss**

기존 BCE loss에 Focal loss, DICE loss, IOU loss를 추가하였습니다.

위의 네가지 loss에 대해 weight를 주어 합하는  Combined loss를 추가하였습니다.

<img src="../assets/img/portfolio/hand_bone/combine_loss_code.png" alt="combine_loss_code" width="300">


학습을 하며 네가지 loss의 경향성을 모두 확인하며 실험하였습니다.

<table>
  <thead>
    <tr>
      <th>Loss Metric</th>
      <th>Loss value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>total loss</td>
      <td>0.06929</td>
    </tr>
    <tr>
      <td>iou loss</td>
      <td>0.04049</td>
    </tr>
    <tr>
      <td>bce loss</td>
      <td>0.00284</td>
    </tr>
    <tr>
      <td>dice loss</td>
      <td>0.02800</td>
    </tr>
    <tr>
      <td>focal loss</td>
      <td>0.00001</td>
    </tr>
    <tr>
      <td>Average Dice</td>
      <td>0.97162</td>
    </tr>
  </tbody>
</table>


  **실험 결과**

  1. 모든 loss를 사용하면 특정 IOU loss는 줄어들지 않는 등의 문제가 생겼습니다.
  2. Focal loss는 weight를 주지 않아도 0에 수렴할 정도로 낮아지는 경향을 보였습니다.
  3. DICE loss는 Average Dice와 역의 유사성을 보여주었습니다.

  **결론**

  - 대회 Metirc인 DICE loss를 사용하며, 클래스 불균형이 없으므로 Focal loss는 굳이 사용하지 않았습니다.
  - DICE + BCE 혹은 DICE + IOU Combined loss를 적용하였습니다.
      - DICE + BCE를 적용시켰을 때 20 epoch만 돌려도 기본 BCE로 70 epoch를 돌렸을 때의 성능이 나왔습니다.
      
      <img src="../assets/img/portfolio/hand_bone/avg_dice_wandb.png" alt="avg_dice_wandb" width="300">

          
      - 또한 Average Dice가 상승한 것을 확인할 수 있었습니다. 전체적으로 상승했으나 특히 손등에서의 겹치는 pisiform이 마의 구간 0.9를 넘기기도 했습니다.
        <table>
          <thead>
            <tr>
              <th>Loss</th>
              <th>pisiform dice</th>
              <th>Average Dice</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>BCE</td>
              <td>0.8900</td>
              <td>0.9700</td>
            </tr>
            <tr>
              <td>DICE + BCE</td>
              <td>0.9088</td>
              <td>0.9713</td>
            </tr>
          </tbody>
        </table>

**Ensemble**

Group KFold를 사용하였으므로 각 fold를 합치는 KFold 앙상블을 사용하였습니다.

Metric이 좋은 두개의 모델에 대해 모델 앙상블을 사용하였습니다.

---
# Skills
<table style="width: 100%; table-layout: fixed; border-collapse: collapse;">
  <tr>
    <th style="width: 40%; padding: 10px; text-align: left; background-color: #f4f4f4;">Hard Skills</th>
    <th style="width: 20%; padding: 10px; text-align: left; background-color: #f4f4f4;">Soft Skills</th>
    <th style="width: 40%; padding: 10px; text-align: left; background-color: #f4f4f4;">Tools</th>
  </tr>
  <tr>
    <td style="padding: 10px; vertical-align: top;">
      <ul>
        <li><b>Python</b><br>Python을 사용하여 다양한 알고리즘을 설계하고 구현할 수 있는 능력을 보유하고 있습니다.</li>
        <li><b>수학 및 통계</b><br>데이터 분석과 인공지능에 필요한 미적분, 확률 및 통계에 대해 이해하고 있습니다.</li>
        <li><b>이미지 분석</b><br>이미지 탐색적 데이터 분석(EDA)을 수행하고, 분석 결과에 따라 적절한 데이터 증강 기법을 적용할 수 있습니다.</li>
      </ul>
    </td>
    <td style="padding: 10px; vertical-align: top;">
      <ul>
        <li>끈기</li>
        <li>꼼꼼함</li>
        <li>시간 관리</li>
      </ul>
    </td>
    <td style="padding: 10px; vertical-align: top;">
      <ul>
        <li>GitHub</li>
        <li>PyTorch</li>
        <li>MMDetection</li>
        <li>MMSegmentation</li>
        <li>OpenCV</li>
        <li>Matplotlib</li>
        <li>Albumentations</li>
      </ul>
    </td>
  </tr>
</table>