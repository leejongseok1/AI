# Object Tracking

- 객체 감지 이후, 그 객체의 움직임을 공간상이나 다양한 카메라 각도에서 추적하는 컴퓨터 비전 응용 기술
- Object Detection: 이미지나 영상 내에서 객체가 어디에 있는지 찾아냄
- Object Tracking: 감지된 객체를 비디오의 여러 프레임에 걸쳐 지속적으로 추적함

<br>

- Object Tracking의 기본 구조
    1. Frame에서 Detection하여 bbox, class 결과를 얻음. 각 객체는 ID를 부여받음
    2. 직전 프레임의 ID를 가진 객체와 새로 탐지된 객체를 비교
        1. 연결기준은 IoU, appearance 등등
        2. 매칭이 성공하면 기존 ID를 물려주고, 매칭이 안되면 새로운 ID를 부여
    3. ID가 부여된 객체들을 프레임마다 이어붙여서 추적
        1. 프레임이 넘어가도 같은 ID를 계속 이어서 기록, 시간에 따른 객체의 이동 경로 추적
        2. 객체가 사라지면 Tracker는 몇 프레임동안 기다리다가 안나타나면 ID 삭제 처리


## SOT, MOT
- Object Tracking은 크게 Single Object Tracking과 Multiple Object Tracking으로 분류됨

### Single Object Tracking

- 영상 내 하나의 객체만 지속적으로 추적하는 문제
- 초기 프레임에서 수동으로 타겟(추적할 객체)을 지정해줌
- 이후 Detetor없이 모델이 객체의 위치를 프레임별로 직접 예측
- 새로운 객체는 신경쓰지 않고 무시함
- 수동으로 지정한 타겟 객체는 외형이 변하거나 가려지거나 스케일이 달라져도 찾을 수 있어야함
- **주요 도전 과제**: 외형 변화, 회전, 스케일 변동, occlusion 등
- **대표 알고리즘**: SiamFC, SiamRPN, SiamMask, DCF계열 등
- SOT 응용 사례/분야
    - 드론이 하나의 사람/차량을 계속 추적해야할 때
    - 스포츠 경기에서 선수 한 명의 움직임을 분석할 때
    - CCTV로 특정 용의자 한 명을 추적할 때

### Multiple Object Tracking

- 영상 내 여러 객체를 동시에 추적하는 문제
- 초기 타겟 수동 지정 없이 detector가 자동으로 프레임마다 객체를 찾음
- 새로 등장하는 객체와 사라지는 객체 모두 관리
- 여러 객체 사이에서 ID가 혼동되지 않게 관리하는 게 중요
- **주요 도전 과제**: 객체 간 유사성, occlusion, ID 스위치, 새 객체 등장, 기존 객체 삭제 처리
- **대표 알고리즘**: SORT, DeepSORT, ByteTrack, FairMOT
- MOT 응용 사례/분야
    - 공항, 지하철 등 군중 모니터링
    - 자율주행 (차량, 보행자 추적)
    - 매장 내 사람들의 움직임 추적

<br>
<br>

## MOT Metrics

### MOTA (Multi-Object Tracking Accuracy)

- 여러 오류를 합쳐 Tracking 정확도를 평가하는 지표

<br>

$$
\text{MOTA} = 1 - \frac{\text{FN} + \text{FP} + \text{IDSW}}{\text{GT}}
$$

- **FN** (False Negative): 놓친 객체 수
- **FP** (False Positive): 잘못 탐지한 객체 수
- **IDSW** (ID switch): ID가 바뀐 횟수
- **GT** (Ground Truth): 전체 정답 객체 수

<br>

- 종합적인 오류를 쉬운 하나로 표현해 한눈에 직관적으로 Tracking 성능을 볼 수 있다.
- Detection 영향에 너무 민감해 Tracking 알고리즘 자체의 성능을 분리해서 보기는 어렵다.
- IDSW를 그냥 오류 하나로만 더하기 때문에 ID를 얼마나 꾸준히 유지했는지를 정교하게 평가하지 못한다.

<br>

### IDF1

- Tracking 중 객체의 ID를 얼마나 일관성 있게 유지했는가 를 평가하는 지표
- F1 Score처럼 Precision과 Recall의 조화평균을 냄냄

<br>

$$
\text{IDF1} = 2 \times \frac{\text{ID Precision} \times \text{ID Recall}}{\text{ID Precision} + \text{ID Recall}}
$$

- ID Precision = 맞게 ID 부여한 수 / 부여한 전체 ID 수
- ID Recall = 맞게 ID 부여한 수 / 실제 GT ID 수

<br>

- ID 일관성을 정확하게 평가함
- 소수의 Detection miss나 FP에는 robust 함.
- Detection이 심각하게 무너지면 IDF1도 같이 하락함

<br>

### MT / ML (Mostly Tracked / Mostly Lost)

- 


<br>

### FAF (False Alarms per Frame)

- 프레임당 평균적으로 얼마나 많은 False Positive가 발생했는지를 측정하는 지표

### MOTP

- 