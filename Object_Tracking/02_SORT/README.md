# Simple Online AND Realtime Tracking (SORT)

- **논문명**: Simple Online AND Realtime Tracking (SORT)
- **저자**: Alex Bewley, Zongyuan Ge, Lionel Ott, Fabio Ramos, Ben Upcroft
- **링크**: [arXiv](https://arxiv.org/pdf/1602.00763)

<br>

----------
<br>

## 3 METHODOLOGY

### 3.1 Detection
- Faster R-CNN 사용
    - PASCAL VOC로 Pre-train된 Faster R-CNN의 기본 파라미터 적용
    - 다른 클래스는 무시하고 보행자(pedestrian)만 탐지함
    - confidence score가 0.5이상인 탐지 결과만 Tracking 프레임워크에 전달

    *prediction: Tracker가 계산한 예상 위치 <br>
    *detection: detector가 탐지한 객체

<br>

### 3.2 Estimation Model
- 각 객체의 프레임 간 이동은 다른 객체나 카메라 움직임과 무관한 선형 상수 속도 모델로 근사
    - 다른 객체나 카메라 움직임은 고려하지 않고 현재 속도 그대로 움직일 거라고 가정
- 탐지 결과가 있을 때 (Detection-Association 성공 시)
    - 탐지 정보로 칼만 필터 보정해 더 정확하게 위치와 속도까지 업데이트
- 탐지 결과가 없을 때 (Detection-Association 실패 시)
    - 기존에 가지고 있던 속도 정보만 이용해서 단순히 예측만 함
    - 탐지 결과가 없으니 내부 모델만 믿고 계속 상태를 이어간다는 의미

<br>

- 하나의 객체(target)를 표현하는 정보
$$ \mathbf{x} = [u, v, s, r, \dot{u}, \dot{v}, \dot{s}]^T $$

- u, v : 중심 좌표 (수평 및 수직 픽셀 좌표)
- s : scale, 영역 면적, 크기
- r : 종횡비 (가로:세로 비율), 상수로 간주
- u', v', s' : 움직이는 속도

#### SORT 순서

1. 1프레임 처리
    1. 1프레임 이미지를 Faster R-CNN 추론 -> 결과물: 객체들의 bounding box 좌표 + confidence score
    2. Association: 1프레임은 아직 매칭 할 대상이 없음
    3. 탐지된 바운딩 박스들에 대해 각각의 새로운 Track ID를 부여
2. 2프레임~
    1. 2프레임 이미지를 Faster R-CNN 추론 -> 결과물: 객체들의 bounding box 좌표 + confidence score
    2. Prediction: 1프레임에서 부여한 Track ID 들의 다음 위치를 칼만 필터를 이용해 예측함
    3. Association: 예측된 위치와 2프레임의 새 탐지 결과의 IoU를 계산해 높은 것들끼리 Hungarian Algorithm을 사용해 최적의 매칭을 찾음
    4. 매칭이 된 Track은 탐지 결과를 이용해 칼만 필터 업데이트 (위치,속도를 정교하게 맞춤)
    5. 매칭이 안 된 Track은 그냥 예측한 위치대로 계속 진행
    6. 매칭이 안 된 탐지 결과는 새롭게 Track ID를 부여해서 Tracking 시작
3. 다음 프레임 부터는 2프레임 과정과 똑같이 반복

> **여기서 내가 궁금했던 부분**: 굳이 예측을 왜 해야되는 거지?
>
> - MOT는 객체에 고유 ID를 부여하고 매 프레임마다 같은 ID를 추적하는 것임
> - 프레임마다 detection 결과가 나오고 기존 타겟과 새로운 탐지를 매칭해야함 
> - 매칭 할 때 IoU 기반으로 겹치는 영역을 계산해서 연결, 예측이 없으면 IoU가 충분히 겹치지 않을 수 있음
> - 예측이 없으면 그냥 프레임마다 detection 결과만 보고 경로를 그리는 것
> - Tracking은 매 프레임마다 '예측 - 비교 - 업데이트'를 반복해서 객체를 Tracking 한다.

<br>

### 3.3 Data Association

- 기존 타겟들에 탐지 결과를 할당할 때 각 타겟의 바운딩 박스는 현재 프레임에서의 새로운 위치를 칼만 필터로 예측
- 그 다음, 할당 비용 행렬은 각 탐지 결과와 기존 타겟의 예측된 바운딩 박스들 간의 IoU 기반으로 계산
- 할당 문제는 Hungarian Algorithm을 이용해 최적으로 해결함
    - Cost = 1 - IoU, 이를 Hungarian Algorithm에 넣는 것
- IoU Threshold를 설정해 더 낮은 경우 연결 X
- IoU 거리를 사용함으로써 지나가는 객체에 의해 발생하는 short-term occlusion 문제를 자연스럽게 처리 가능
- 타겟이 다른 객체에 의해 가려질 경우, 가리는 객체만 탐지되는 상황이 발생하는데 IoU 거리는 스케일이 비슷한 탐지를 우선시하기 때문에 가리는 객체는 탐지를 통해 보정되고 가려진 객체는 연결되지 않아 영향을 받지 않게 된다.

> **여기서 내가 궁금했던 부분**: IoU 기반으로 매칭한다고 했는데 만약 크기가 거의 완전 비슷?일치? 한 바운딩박스가 여러 개 있으면 나중 프레임에서 계산할 때 엇갈릴 수도 있지 않나?
>
> - 맞다. 비슷한 IoU 값을 가진 바운딩 박스가 여러 개 있으면 헝가리안 알고리즘이 잘못된 매칭을 할 수도 있고 이 문제를 **ID switch**라고 한다.
> - **ID switch**: 프레임이 넘어가면서 객체의 ID가 바뀌는 현상
> - 그래서 이 문제를 보완한 Tracking 모델은 appearance 특징을 같이 고려한다.
>   - 객체의 색깔, 텍스처, CNN feature 등

<br>

### 3.4 Creation and Deletion of Track Identities

1. Track 생성
    - 새로운 탐지가 기존 트랙과 IoU THR을 충족하지 않으면 새로운 객체라고 판단 후 새로운 Track ID를 부여
    - 이렇게 만들어진 Track ID는 바로 확정되는 게 아니고 '검증 기간'을 거침
        - 일정 프레임 이상 계속 탐지 결과와 연결되어야 한다.
        - 그래야 False Positive(FP)를 방지할 수 있다.
2. Track 삭제
    - 어떤 Track ID가 TLost 프레임동안 탐지되지 않으면 그 Track을 삭제
    - 논문에서는 TLost = 1로 설정 -> 한 번 탐지 실패하면 바로 삭제
    - 잃어버린 객체를 빨리 지우면 Tracker 수가 줄어들어 효율성 증가
    - 만약 삭제된 객체가 다시 나타나면, 새로운 ID를 부여해 다시 Tracking 시작