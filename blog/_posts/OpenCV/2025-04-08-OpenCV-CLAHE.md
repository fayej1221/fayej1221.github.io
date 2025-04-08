---
layout: post
title: CLAHE란?
description: CLAHE
categories: [OpenCV]
tags: [CLAHE]
---

* this unordered seed list will be replaced by the toc
{:toc}

# Histogram Equalization
의료 영상은 포맷 외에도 조명 및 대비 문제로 인해 시각적 이해가 어려움

개선하기 위해서 히스토그램 평활화 과정 적용

주로 이미지의 대비를 향상시키기 위해 사용

opencv에서 cv2.equalizeHist를 통해 가능

# CLAHE(Contrast Limited Adaptive Histogram Equalization)

Histogram Equalization은 원본 전체 이미지에 적용

CLAHE는 이미지를 작은 블록으로 나누고 히스토그램 평활화를 개별적으로 진행하여 노이즈를 과도하게 증폭하지 않으면서 지역 대비를 향상시킬 수 있음

CLAHE는 과도한 대비 증폭 제한을 위해서 파라미터 사용

opencv에서 createCLAHE를 통해 가능
- clipLimit: contrast 제한을 위한 threshold
- tileGridSize: 히스토그램 평활화를 위한 grid 사이즈

