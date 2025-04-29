# Simple Online and Realtime Tracking with a Deep Association Metric

- **논문명**: Simple Online and Realtime Tracking with a Deep Association Metric
- **저자**: Nicolai Wojke†, Alex Bewley, Dietrich Paulus
- **링크**: [arXiv](https://arxiv.org/pdf/1703.07402)

<br>

----------
<br>

- 객체의 appearance(외형) 정보를 활용해 SORT의 ID switch 문제를 줄이고 occlusion 상황에서도 안정적으로 Tracking할 수 있도록 개선한 실시간 Tracking 알고리즘

![Image](https://github.com/user-attachments/assets/de9d8c5b-0f3f-439d-b4b6-cd4392f76bf6)

## 2 SORT WITH DEEP ASSOCIATION METRIC

### 2.1 Track Handling and State Estimation

- 기존 SORT와 동일한 칼만 필터 기반 프레임워크를 사용
- Tracking 시나리오 가정
    - 카메라는 calibration 되어있지 않다.
    - Ego-motion 정보(카메라 자체의 이동 정보)는 사용할 수 없다.

- 상태 공간
    - 객체는 8차원 상태 벡터로 표현됨 (SORT는 6차원)
    - $$ (u, v, \gamma, h, \dot{u}, \dot{v}, \dot{\gamma}, \dot{h}) $$
    - u, v : 바운딩 박스 중심 x, y 좌표
    - r : 종횡비
    - h : 바운딩 박스 높이
    - u', v', r', h' : 각각의 속도 (1차 미분값)

    - 오래 매칭되지 않는 Track은 삭제, 새로운 객체는 새로운 Track으로 초기화하되, 3 프레임 안에 검증 실패하면 폐기(잠정적 트랙 처리 -> FP를 걸러내기 위함)
    - 실시간성이 매우 중요하기 때문에 복잡한 트랙 관리 대신 간결한 트랙 유지/삭제 방식을 취함

<br>

### 2.2 Assignment Problem

- 이전 프레임에서 칼만 필터로 예측한 객체 위치들과, 현재 프레임에서 탐지된 객체 결과들을 어떻게 연결할지 결정하는 문제를, 헝가리안 알고리즘으로 푸는 '할당 문제(assignment problem)'로 정의하는 것이 전통적인 방식이다.
- DeepSORT 에서는 여기에 motion information과 appearance information을 결합한 metric을 통합한다.

<br>

#### 1. Motion information: Mahalanobis Distance
- association을 할 때 motion information을 반영하는 방법으로 마할라노비스 거리 사용
- **마할라노비스 거리**
    - 두 확률 분포 또는 점 간의 거리 척도로써 분포의 공분산을 고려해 정규화된 거리를 게산하는 방법
    - 예측된 칼만 상태 (y_i, S_i) (평균위치, 공분산)와
    - 새로운 측정값 d_j 간의 마할라노비스 거리를 정의
        $$ (^{(1)}(i,j) = (d_j - y_i)^T S_i^{-1} (d_j - y_i)) $$

    - **d_j** : j번째 탐지 결과 (바운딩 박스)
    - **y_i** : i번째 트랙의 칼만 필터 예측 위치
    - **S_i** : i번째 트랙의 예측 공분산 행렬
    - => : **"탐지 결과 d_j가 예측된 트랙 위치 y_i로부터 몇 표준편차 떨어져 있는가?"**


    - 칼만 필터는 각 트랙의 예측 위치뿐만 아니라 그 위치에 대한 불확실성(공분산행렬)도 제공하는데 단순 거리로는 판단하기 어려운 예측 오차 범위를 고려해 탐지 결과가 얼마나 예측에 부합하는지 평가 가능
    - 마할라노비스 거리가 일정 Threshold 이하인 경우에만 매칭 가능하다고 판정
    - d_(i,j)가 THR 이하이면 매칭 가능, 초과이면 매칭 불가
        - THR은 카이제곱 분포에 의해 결정됨 -> t = 9.4877
    - 마할라노비스 거리를 사용하는 이유
        - 예측 위치에서 너무 멀리 떨어진 탐지 결과는 고려 X -> 잘못된 매칭을 방지
        - 칼만 필터의 예측 결과와 불확실성을 활용해 현실적으로 가능한 트랙-탐지 매칭 후보만 남길 수 있음
    - 마할라노비스 거리의 한계
        - 카메라 움직임이 존재하거나 예측 오차가 커지면 신뢰성을 잃음
        - occlusion 상황에서는 이 거리만으로 객체를 제대로 연결하기 어려움 -> appearance를 사용한 cosine distance 도입

<br>

#### 2. Appearance information: cosine distance

- appearance descriptor
    - 각 탐지가 결과 바운딩 박스 d_j에 대해 CNN 모델을 사용해 정규화된 벡터 r_j를 생성함
    - r_j가 객체의 외형을 요약한 feature vector

- appearance gallery
    - 각 트랙 k(추적 중인 객체)에 대해서 이 트랙과 association된 과거 appearance descriptor들을 최대 L_k = 100 개 까지 저장, 이 모음을 R_k = {r_k^(i)} 라 표기
    - 트랙 k가 과거 프레임에서 가졌던 appearance 정보들을 저장하는 것

- cosine distance
    - 트랙 k와 새로 들어온 탐지 결과 d_j간의 appearance similarity는 가장 가까운 appearance descriptor 쌍을 찾아서 계산함
        $$ d^{(2)}(i,j) = \min \left\{ 1 - r_j^T r_k^{(i)} \mid r_k^{(i)} \in R_k \right\} $$

    - cosine similarity는 "벡터 방향이 얼마나 비슷한가"를 보는 값이기 때문에 1 - cosine similarity를 취해 거리를 만든다
    - 1 - cosine similarity 값이 작을수록 (cosine similarity가 클수록) 두 객체는 외형적으로 비슷하다고 판단
    - appearance distance d^(2)(i,j)가 THR 이하이면 트랙 i와 탐지 j간의 association을 허용

- CNN Model
    - person Re-ID 데이터셋에서 학습한 모델 사용
    - 2개의 conv와 6개의 residual block
    - dense layer 10에서 차원이 128인 global feature map을 계산
    - Batch Norm과 l2 Norm 사용 (l2 정규화는 벡터 크기를 1로 맞춤)
    - 이렇게 해서 cosine appearance metric과 호환되게 만듦
    - CNN이 출력한 feature vector끼리 cosine distance를 계산
    - ![Image](https://github.com/user-attachments/assets/3ade2ee9-c7bc-4255-be95-49b45c4a88e9)

<br>

### 2.3 Matching Cascade

- SORT에서는 모든 Track과 모든 detection 결과를 한 번에 매칭하는 global assignment problem을 풀고 이 문제를 hungarian algorithm으로 최적으로 해결했지만 몇 가지 문제점이 존재한다.
    1. 칼만 필터 불확실성 증가
        - 칼만 필터는 매 프레임마다 다음 위치를 예측한다.
        - 탐지가 없으면 업데이트가 안되고 예측만 반복되기 때문에 **상태 추정의 불확실성이 게속해서 증가한다.**
        - 즉, 예측 위치가 점점 퍼지고 범위가 넓어지는 것이다.
    2. 마할라노비스 거리 문제
        - 마할라노비스 거리는 "예측 오차 범위 내 표준편차 거리"를 측정하는데
        - 오차 범위(공분산)가 커지면, 같은 거리라도 표준편차 단위로는 작아 보인다.
        - 그래서 불확실성이 큰 Track이 같은 탐지 결과에 대해 더 짧은 거리를 가지게 된다.
        - 즉 최근에 본 Track보다, 불확실성이 큰 오래된 Track이 해당 바운딩 박스를 매칭해서 점유한다.
    - 그래서 이를 해결 하기 위해 **Matching Cascade**를 도입했다.

<br>

Matching Cascade의 핵심 아이디어는 더 최근에 관측된(탐지된) 트랙에게 먼저 매칭 기회를 주는 것이다.

**각 트랙에 Age를 부여해서** 가장 최근에 탐지된 트랙부터 순서대로 아직 매칭되지 않은 탐지 결과들과 매칭을 시도한다.

#### Matching Cascade Algorithm

![Image](https://github.com/user-attachments/assets/932a7cd4-68c4-4024-b74e-24d9b1c35483)

- Input: Track Index 집합 **T**, detection Index 집합 D, 최대연령 A_max
- Age: 그 트랙이 마지막으로 탐지 결과와 매칭된 이후 지난 프레임 수, 프레임 마다 Age+=1

과정:

1. 전체 Track과 Detection 간 cost matrix와 허용 여부 행렬(admissible matrix)를 계산
2. Track의 Age n을 1부터 A_max 까지 증가시키면서 반복
    1. 최근 n 프레임 동안 매칭되지 않은 Track들 T_n을 선택
        1. ex) Age가 1인 트랙들 T_1만 모음
    2. T_n과 아직 매칭되지 않은 탐지 결과 U 간에 선형 할당 문제 풀이
        1. ex) T_1과 U 사이에서 헝가리안 알고리즘으로 매칭
    3. 매칭된 쌍을 result에 추가하고, 매칭된 detection을 U에서 제거
    4. 매칭 결과와 남은 unmatched detection들을 반환

and,

급격한 외형 변화를 보완하고, 잘못된 Track 초기화를 바로잡기 위해서 아직 확정하지 않은 Track이나 n=1 프레임 동안만 관측된 트랙에 대해 IoU 기반의 추가 매칭 적용 (SORT에서의 방법과 동일)

<br>

## 3 EXPERIMENTS

![Image](https://github.com/user-attachments/assets/6657fde4-6fa0-41f9-8a1f-dc59ed156386)

<br>

## 4 CONCLUSION

DeepSORT는 SORT에 appearance 정보를 통합하여 longer periods of occlusion 상황에서도 안정적으로 객체를 추적할 수 있게 했고, 여전히 구현이 간단하고 실시간으로 동작한다.