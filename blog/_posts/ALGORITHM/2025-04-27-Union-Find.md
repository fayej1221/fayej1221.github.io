---
layout: post
title: Union Find란?
description: > 
  Union Find 설명과 기본 구현, 최적화 구현
categories: [ALGORITHM]
tags: [Disjoint Set, Union Find]
---
[Union-Find 알고리즘](https://gmlwjd9405.github.io/2018/08/31/algorithm-union-find.html)
[Disjoint Set #2 Union By Rank와 Path Compression](https://yonghwankim-dev.tistory.com/236)
{:.note title="출처 및 참고"}

* this unordered seed list will be replaced by the toc
{:toc}

# Disjoint Set
**서로 중복되지 않는 부분 집합**들로 나눠진 원소들에 대한 정보를 저장하고 조작하는 자료 구조

- 공통 원소가 없는, 즉 상호 배타적인 부분 집합들로 나눠진 원소들에 대한 자료구조
- 서로소 집합 자료구조

# Union-Find
Disjoint Set을 표현할 때 사용하는 알고리즘
- 집합을 구현하는 데는 비트 벡터, 배열, 연결 리스트를 이용할 수 있으나 그 중 **효율적인 트리구조를 이용하여 구현**(배열의 경우 union을 할 때 다 순회해서 집합 번호 변경해야하지만 트리에서는 루트 노드만 찾아서 연결)

## 연산
1. make-set(x): 초기화, x를 유일한 원소로하는 새로운 집합을 만듦
2. union(x, y): 합하기, x가 속한 집합과 y가 속한 집합을 합침
3. find(x): 찾기, x가 속한 집합의 대표값(루트 노드 값)을 반환

## 사용 예시
- Kruskal MST에서 새로 추가할 간선의 양끝 정점이 같은 집합에 속해 있는지(사이클 형성 여부 확인)에 대해 검사하는 경우
- 초기에 n+1개의 집합을 이루고 있을 때, 합집합과 두 원소가 같은 집합에 포함되어 있는지를 확인하는 연산을 수행하려는 경우 - 백준: 1717
- 어떤 사이트의 친구 관계가 생긴 순서대로 주어졌을 때, 가입한 두 사람의 친구 네트워크에 몇 명이 있는지 구하는 프로그램 - 백준: 4195

## 기본 구현
```python
root = [i for i in range(MAX_SIZE)]

def find(x):
    if root[x] = x:
        return x
    else:
        return find(root[x])

def union(x, y):
    x = find(x)
    y = find(y)
    root[y] = x
```

## 최적화한 구현 방법
재귀적인 호출로 부모를 타고 올라가는 방식인 **find 연산은 연결 리스트처럼 되므로 높이가 h면 최악의 경우 O(h)**

### union-by-rank(union-by-height)
**2개의 부분 집합이 서로 병합일 때 더 작은 트리가 더 큰 트리 자식으로 병합**, 둘을 비교하기 위해서 rank를 쓰는데, height와는 다름

1. union 호출 전 각각 부분집합의 루트를 find하고 rank 비교
2. 두 집합의 rank가 동일하면 왼쪽이 오른쪽의 자식이 되고 rank 1 증가
3. 그 외에는 rank가 낮은 쪽의 트리가 더 큰 쪽의 트리의 자식이 됨

### 경로 압축(Path Compression)
find 연산 호출시 트리의 높이를 평평하게 하는 것

어떤 요소 x에 대해 find 호출시 트리의 루트를 반환하게 함, find 연산을 호출하면 어떤 요소 x에서 x가 속한 트리의 루트까지 이동함

**탐색된 루트 노드를 x의 부모로 만들어** 모든 중간 노드를 이동할 필요가 없게 만듦

## 최적화 코드, 시간 복잡도
두 기술을 적용한 find는 $O(log_2*N)$

```python
root = [i for i in range(MAX_SIZE)]  # 부모 배열
rank = [0 for _ in range(MAX_SIZE)]  # 트리 높이를 저장하는 배열

def find(x):
    if root[x] != x:
        root[x] = find(root[x])  # 경로 압축
    return root[x]

def union(x, y):
    rootX = find(x)
    rootY = find(y)

    if rootX != rootY: # 서로 다른 집합에 속할 때만 합침(rank 기준)
        # y가 더 크면
        if rank[rootX] < rank[rootY]: 
            root[rootX] = rootY
        # x가 더 크면
        elif rank[rootX] > rank[rootY]:
            root[rootY] = rootX
        else:
            root[rootY] = rootX
            rank[rootX] += 1
```

union-find의 연산 복잡도는 $O(N+Mlog_2*N)$, N은 노드의 개수, M은 에지의 개수

하지만 **상수는 제거되고 상수 시간을 가져 $O(1)$**

# 분할(Partition)
임의의 집합을 분할한다는 것은 각 부분 집합이 아래의 두 가지 조건을 만족하는 Disjoint Set이 되도록 쪼개는 것

1. 분할된 부분 집합을 합치면 원래의 전체 집합이 됨
2. 분할된 부분 집합끼리는 겹치는 원소가 없음