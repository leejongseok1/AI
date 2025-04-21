# 01_Vision_Transformer

- **논문명**: An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale
- **저자**: Dosovitskiy et al. (Google Research, 2020)
- **링크**: [arXiv](https://arxiv.org/pdf/2010.11929)

<br>

----------
<br>

## 1 Introduction

- 여러 연구에서 self-attention을 CNN과 결합하려는 시도를 많이 했고, 일부 연구는 convolution 연산을 완전히 대체하기 위한 시도를 하였다.
- 이론적으로는 효율적이지만, 특수한 attention 패턴을 사용하기 때문에 현대 GPU 상에서는 효과적이지 못하였다.
- 최소한의 수정만으로 Transformer를 이미지에 적용해보았음
- 이미지를 patch로 나눈 뒤, 각 patch를 linear embedding한 시퀀스를 Transformer에 input으로 넣음
    - 여기서 이미지 patch가 NLP에서의 token과 동일한 방식으로 취급된다.
- 그 결과, ResNet보다 조금 낮은 정확도를 보였다. Transformer는 CNN이 지닌 고유의 inductive bias (translation equivariance, locality)를 갖추고 있지 않기 때문에, 적은 양의 데이터로는 일반화 성능이 떨어진다.
- 하지만, 대규모 데이터셋 (1400만~3억 장)으로 학습할 경우에는 귀납적 바이어스(inductive bias)를 능가하였다.
- ViT는 충분한 규모로 Pre-train하고, 이후 데이터가 적은 task에 transfer learning했을 때 매우 뛰어난 성능을 보인다.
- ViT는 여러 이미지 인식 벤치마크에서 기존의 SOTA에 접근하거나 뛰어넘었다.

## 2 Related Work

- Transformer 기반 모델은 대규모 말뭉치에 대해 사전학습 후, 파인튜닝된다.
    - BERT: 노이즈 제거 자기 지도 학습 과제를 사전학습
    - GPT: 언어 모델링을 사전학습 과제로 사용

- Naive Application
    - Self-attention을 이미지에 적용하면 각 픽셀이 다른 모든 픽셀을 attend해야 하므로, 픽셀 수의 제곱에 비례하는 계산 비용이 발생하기 때문에 적용이 어렵다.
    - parmar: 전역이 아닌 각 쿼리 픽셀에 대해 국소 영역(local neighborhood)에만 self-attention 적용
    - Sparse Transformer, Weissenborn: 다양한 크기의 block 단위로 scale attention, 효율적인 GPU 사용을 위한 복잡한 엔지니어링이 필요
    - Cordonnier: ViT와 가장 유사한 모델.
        - 이미지에서 2x2 patch를 추출하고 그 위에 완전한 self-attention을 적용

    - Image GPT(iGPT): 이미지 해상도와 색상 공간을 축소한 후, Transformer를 픽셀에 적용. 비지도학습


## 3 Method

![Image](https://github.com/user-attachments/assets/2f04b37a-e7a6-47fb-b771-aed0e8123226)

### 3.1 Vision Trnasofmer (ViT)

1. 이미지를 patch로 나눔
    - 224x224 이미지를 16x16 블록으로 나눔 -> 총 196개의 patch 생성
    - 각 patch는 16x16x3 = 768 Demension

2. 각 patch를 vector로 바꿈
    - 각 patch를 flatten하여 하나의 vector로 만듦
    - 학습 가능한 선형 변환을 통해 Transformer가 이해할 수 있는 고정된 차원 D로 매핑 (718 -> 512)
    - 이 투영 결과를 **patch embedding** 이라고 함

3. patch embedding sequence 앞에 [CLS] Token을을 추가
    - BERT의 [class] 토큰과 유사하게 [class] 토큰을 앞에 붙여줌줌
    - 이는 Transformer가 최종적으로 이 이미지가 어떤 클래스인지 판단하는 데 사용

4. Positional Embedding 추가
    - 이 patch가 어디에 있었는지 알려주기 위해 Positional Embedding을 더해줌으로써 위치 정보를 알려줌

5. Transformer Encoder
    - 196개의 patch와 1개의 class token, 총 197개의 벡터를 Transformer Encoder에 Input으로 넣음
    - Multi-Head Self Attention 적용
    - MLP (Feedforward Network)
    - 모든 Block 이전에 LayerNormalization(LN) 적용
    - Residual Connection을 모든 블록 마지막에 적용
    - 위 과정을 N-layer번 반복

6. 최종적으로 Transformer 마지막 층의 [CLS] token output만 뽑아서 MLP를 통과시켜 분류 결과를 얻음

<br>

### 3.2 Fine-Tuning and Higher Resolution

- ViT는 보통 대규모 데이터셋에서 사전학습하고, 이후에 작은 downstream task들에 fine-tuning 하는 방식으로 활용된다.
- 사전학습한 ViT에는 원래 이미지 분류를 위한 Classification Head가 달려있는데, 이것을 제거하고 새로 초기값이 0인 D * K 크기의 feedforward를 붙인다. (D: Transformer의 출력 벡터 크기, K: 다운스트림 작업의 클래스 수)

<br>

- fine-tuning을 사전학습보다 더 높은 해상도로 수행하면 성능이 더 좋아질 수 있다.
- 동일한 patch 크기를 유지한다면 결과적으로 더 많은 patch가 생기고 입력 시퀀스 길이가 늘어날 것이고 ViT는 메모리 한도 내에서는 시퀀스 길이가 늘어나도 상관이 없다.
- 하지만 문제는 Positional Embedding이다. 사전학습 시에는 기존 패치 기준의 위치 임베딩만 학습되어 있는데 고해상도 이미지를 넣으면 패치 수가 달라지기 때문에 기존 위치 임베딩을 그대로 쓸 수 없다.
- 이를 해결하기 위해 ViT는 위치 임베딩을 2차원 공간 상에서 interpolation 한다.
- 바로 이 부분이 ViT에 대해 이미지의 2D 구조에 관한 inductive bias가 수동으로 주입되는 유일한 지점이다.