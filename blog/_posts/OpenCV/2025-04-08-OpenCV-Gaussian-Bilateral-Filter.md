---
layout: post
title: GaussianFilter, BilateralFilter?
description: GaussianFilter와 BilateralFilter
categories: [OpenCV]
tags: [GaussianFilter, BilateralFilter]
---

[계산사진학 Edge Aware Image Filtering - Bilateral Filtering](https://velog.io/@claude_ssim/%EA%B3%84%EC%82%B0%EC%82%AC%EC%A7%84%ED%95%99-Edge-Aware-Image-Filtering-Bilateral-Filtering)

[영상처리 5.5 Bilateral Filtering](https://velog.io/@jungizz_/%EC%98%81%EC%83%81%EC%B2%98%EB%A6%AC-5.5-Bilateral-Filtering)

{:.note title="출처 및 참고"}

이후에 또 보게 되면 위의 블로그 두 개를 참고

* this unordered seed list will be replaced by the toc
{:toc}

# Edge Aware Filter
edge aware filter: 작은 디테일들을 smooth하게 만들면서 구조적인 edge들은 보존하는 filter

task: noise를 제거하도 edge를 보존

보통 noise는 high frequency componet가 대부분

high frequency componet를 제거하기 위해서 lowpass filter를 주로 사용했지만 그냥 lowpass filter를 사용하면 전체가 블러리해짐

-> noise만 제거하고 edge 보존하기 위해서 linear filter나 Gaussian filter를 사용하면 noise가 제거될 수 있어도 전체적인 이미지는 블러리하게 될 것 그렇기에 edge aware filter를 사용

# Gaussian Filtering 문제점
Gaussian filter를 적용할 때 이미지의 각 픽셀마다 작은 window를 crop하여 Gaussian kernel과 dot product를 진행

이때 모든 픽셀에 Gaussian kernel를 사용하게 됨

Edge가 심하게 존재하는 patch에 대해서 Gaussian kernel을 곱하게 되면 차이가 드러나는 intensity 값이 혼합되어 edge 마저도 smooth하게 됨

# Bilateral Filtering

edge를 보존하여 noise를 제거하기 위해서 edge aware filter의 일종인 bilateral filter를 사용

이미지 patch의 특성을 고려해서 kernel의 모양을 다르게 하여 filtering을 진행

# Gaussian Filtering VS Bilateral Filtering

**spatial weighting function**: target 픽셀 부근에서 큰 비중으로 이뤄짐. **target 픽셀에 가까울수록 큰 값을 곱함**

Gaussian은 spatial weighting function이 모든 픽셀과 곱함
Gaussian의 목적은 주변의 픽셀과 비슷하도록 값을 blend해주는 것임

**intensity range weighting function**: target 픽셀과 **비슷한 intensity 값을 가지는 픽셀에 더 큰 weight를 부여**

-> output 이미지 픽셀은 input 이미지 픽셀과 비슷한 intensity를 가지게 되면 더 큰 영향을 주게 됨

> **Gaussian filtering** Smooths everything nerby (even edges) Only depends on spatial distance

> **Bilateral filtering** Smooths 'close' pixels in space and intensity depends on spatial and intensity distance

## bilateral filter가 모든 edge를 고려?
Intensity range weighting function은 단순하게 intensity 값을 통해서 계산

비록 edge로 분리가 되어있다고 하더라도 비슷한 intensity 값을 가지는 픽셀은 여전히 같이 blend

# Joint Bilateral Filter: depth map에서도 사용
Intensity range weighting function이 원래의 이미지가 아닌 빛에 의해 촬영이 된 guide 이미지로부터 계산이 되도록 수정

-> 더 정확한 intensity range를 계산하게 되어 edge를 보존하게 됨

depth map denoising에도 사용됨, Conventional depth 카메라는 많은 noise를 가지며 제거하기 위해서 사용

# OpenCV
> cv2.bilateralFilter(src, d, sigmaColor, sigmaSpace, dst=None, borderType=None) -> dst