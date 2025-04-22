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

### 3.1 Vision Transformer (ViT)

1. 이미지를 **patch로 나눔**
    - 224x224 이미지를 16x16 블록으로 나눔 -> 총 196개의 patch 생성
    - 각 patch는 16x16x3 = 768 dimension

2. 각 patch를 **embedding vector로 변환**
    - 각 patch를 flatten하여 하나의 vector로 만듦
    - 학습 가능한 선형 변환을 통해 Transformer가 이해할 수 있는 고정된 차원 D로 매핑 (768 -> D)
    - 이 투영 결과를 **patch embedding** 이라고 함

3. patch embedding sequence 앞에 **[CLS] Token을 추가**
    - BERT의 [class] 토큰과 유사하게 [class] 토큰을 앞에 붙여줌
    - 이는 Transformer가 최종적으로 이 이미지가 어떤 클래스인지 판단하는 데 사용

4. **Positional Embedding** 추가
    - 각 patch의 위치 정보를 알려주기 위해 positional encoding을 더함

5. Transformer Encoder에 입력
    - 196개의 patch와 1개의 class token, 총 197개의 벡터를 Transformer Encoder에 Input으로 넣음
        - Multi-Head Self Attention 적용
        - MLP (Feedforward Network)
        - 모든 Block 이전에 LayerNormalization(LN) 적용
        - Residual Connection을 모든 블록 마지막에 적용
        - 위 과정을 N개의 Transformer Layer에 반복 적용

6. 최종적으로 Transformer 마지막 층의 [CLS] token output만 뽑아서 MLP를 통과시켜 분류 결과를 얻음

![Image](https://github.com/user-attachments/assets/46f2ca6d-d934-4495-b52d-74db6a328858)

<br>

### 3.2 Fine-Tuning and Higher Resolution

- ViT는 보통 대규모 데이터셋에서 사전학습하고, 이후에 작은 downstream task들에 fine-tuning 하는 방식으로 활용된다.
- 사전학습한 ViT에는 원래 이미지 분류를 위한 Classification Head가 달려있는데, 이것을 제거하고 새로 초기값이 0인 D * K 크기의 feedforward layer를 붙인다. (D: Transformer의 출력 벡터 크기, K: 다운스트림 작업의 클래스 수)

<br>

- fine-tuning을 사전학습보다 더 높은 해상도로 수행하면 성능이 더 좋아질 수 있다.
- 동일한 patch 크기를 유지한다면 결과적으로 더 많은 patch가 생기고 입력 시퀀스 길이가 늘어날 것이지만, Transformer는 시퀀스 길이가 늘어나도 메모리 한도 내에서는 문제없이 처리 가능하다.
- 하지만 문제는 Positional Embedding이다. 사전학습 시에는 기존 패치 기준의 위치 임베딩만 학습되어 있는데 고해상도 이미지를 넣으면 패치 수가 달라지기 때문에 기존 위치 임베딩을 그대로 쓸 수 없다.
- 이를 해결하기 위해 ViT는 위치 임베딩을 2차원 공간 상에서 interpolation 한다.
- 바로 이 부분이 ViT에 대해 이미지의 2D 구조에 관한 inductive bias가 수동으로 주입되는 유일한 지점이다.

<br>
<br>

## 4 Experiments

- 아래 데이터셋으로 **사전학습**
    - **ILSVRC-2012 ImageNet**: 1,000 classes, 1.3M images
    - **ImageNet-21k**: 21,000 classes, 14M images
    - **JFT**: 18,000 classes, 303M high-resolution images
- 벤치마크 데이터셋: ImageNet, ImageNet-ReaL, CIFAR-10, CIFAR-100, Oxford-IIIT Pets, Oxford Flowers-102, VTAB(Natural, Specialized, Structured)

- ViT 구성을 BERT 설정 기반으로 함. Base, Large는 BERT에서 가져오고 Huge는 따로 추가

![Image](https://github.com/user-attachments/assets/48108bc2-a885-469e-ab76-062755fd6a91)

- 성능 비교
    - ViT (Vision Transformer)
    - ResNet의 Batch Normalization을 Group Normalization으로 바꾼 모델 (BiT)
    - 하이브리드 모델

- Training
    - optimizer: Adam (B_1=0.9, B_2=0.999)
    - batch: 4096
    - weight decay: 0.1
    - 세부사항은 부록에!

- Fine-Tuning
    - optimizer: SGD with momentum
    - batch: 512
    - 세부사항은 부록에!

- Metrics
    - Fine-Tuning accuracy: 각 모델을 해당 데이터셋에 대해 fine-tuning한 후의 정확도
    - Few-shot accuracy: frozen된 feature를 사용해, linear probe로 정규화된 최소제곱 회귀 문제를 풀어 계산한 정확도
    - 주로 Fine-Tuning acc 성능에 초점을 맞추지만, Fine-Tuning 비용이 많이 들면 선형 few-shot acc를 사용해 빠르게 성능 평가

- Results
    - ![Image](https://github.com/user-attachments/assets/d0aac67b-3333-4009-8485-679c5e3a2e8e)
    - ViT-L/16은 JFT-300M에서 사전학습했을 때
        - BiT-L보다 모든 task에서 더 좋은 성능을 보였고 계산량도 훨씬 적음
    - ViT-H/14는 ImageNet, CIFAR-100, VTAB 같은 어려운 데이터셋에서 우수한 성능을 보임
    - ImageNet-21k로 사전학습한 ViT도 높은 성능을 보였고 일반 TPU(8 core)에서도 약 30일 내 학습 가능
    - ViT-H/14는 VTAB의 Natural과 Structured 작업에서 기존 최고 성능(BiT, VIVI, S4L 등)을 능가했으며, Specialized 작업에서는 상위 모델들과 유사한 성능을 보임

### INSPECTING VISION TRANSFORMER

![Image](https://github.com/user-attachments/assets/64c72687-3dc8-460c-908f-7a9ba2421df9)

- patch embedding
    - ViT의 첫 번째 layer는 flattened image patch를 저차원 공간으로 선형 투영한다.
    - 위 사진에서 왼쪽 그림은 학습된 embedding 필터의 주성분인데, 이 성분들은 각 patch 내부의 세밀한 구조를 저차원으로 표현하기 위한 가능성 있는 basis functions 처럼 보인다고 한다.

- position embedding
    - 위 사진에서 가운데 그림은 모델이 이미지 내 distance를 embedding 유사도(similarty)를 통해 인코딩하는 방식을 보여준다. 즉, 더 가까운 패치일수록 더 유사한 position embedding을 갖는다.
    - 또한, 같은 행이나 열에 있는 patch들은 비슷한 embedding을 갖는 경향이 있고
    - 그리드가 커질수록 사인파(sinusoidal) 구조가 나타나기도 한다.
    - 이 결과들은 position embedding이 2D 이미지의 위상(topology)을 학습하게 된다는 것을 의미한다.

- Self-Attention의 거리 특성
    - Self-attention 덕분에 ViT는 가장 낮은 층에서도 이미지 전체에 걸쳐 정보를 통합할 수 있다.
    - attention weight를 기반으로 모델이 얼마나 먼 거리까지 정보를 통합하는지에 대한 **평균 거리**(attention distance)를 계산하였고, 이는 CNN에서의 receptive field와 유사한 개념이다.
    - 결과:
        - 일부 attention head는 가장 낮은 계층에서도 이미지 전체에 걸쳐 주목(attend)하는데 이는 ViT가 전역 정보 통합 능력을 실제로 활용하고 있다는 것을 보여주는 것이다.
        - 다른 attention head들은 낮은 층에서 지속적으로 작은 attention 거리만 갖는데, 이런 국소적 attention은 Transformer 앞에 ResNet을 사용하는 하이브리드 모델에서는 덜 두드러지고 이 attention head들이 CNN의 초기 합성곱 층과 유사한 역할을 한다는 것을 시사한다.
    - 또한 네트워크 깊이가 깊어질수록 attention distance는 증가한다.


<br>

## 5 Conclusion

- 기존 컴퓨터 비전 분야의 self-attention 연구들과는 다르게 초기 패치 분할 단계 외에는 이미지 특화된 inductive bias를 도입하지 않았다.
- 대신 이미지를 patch들의 시퀀스로 해석하고, 이를 표준 Transformer Encoder로 처리한다.
- 대규모 데이터셋으로 사전학습했을 때 놀라운 성능을 보였고, SOTA와 동등하거나 그 이상을 달성했다.
- 사전학습 비용이 상대적으로 저렴하다.

### 도전과제
- ViT를 다른 컴퓨터 비전 과제(object detection, segmentation, ..)에 적용하기
- 지속적인 self-supervised pre-training 연구
- Vision Transformer를 더 크게 확장하기