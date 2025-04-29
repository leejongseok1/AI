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