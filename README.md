# DINOv3 Crop Disease Analysis & Domain Shift

DINOv3 임베딩 모델을 활용하여 농작물(잎) 이미지의 도메인 시프트(Domain Shift) 현상 및 질병 유무(Healthy vs Diseased)를 Centroid Cosine Distance와 차원 축소(t-SNE, UMAP) 기술을 통해 분석하고 분류 모델을 학습하는 저장소입니다.

---

## 0. 외부 종속성 (DINOv3) 설치 및 Git Pull 안내

본 저장소의 핵심 학습 및 분석 기능은 Facebook Research의 **DINOv3 모델**을 참조합니다. 의존성 관리를 위해 `dinov3/` 디렉토리는 본 저장소의 Git 추적 대상에서 제외되어 있으므로, 처음 설치하거나 코드를 최신 상태로 유지하기 위해 아래 명령어를 수행해 주셔야 합니다.

### ① DINOv3 저장소 최초 복제 (Clone)
프로젝트 루트 경로(`DINO3`)에서 아래 명령어를 실행하여 DINOv3 소스 코드를 복제합니다.
```bash
git clone https://github.com/facebookresearch/dinov3.git
```

### ② 최신 코드 업데이트 (Git Pull)
DINOv3 모델 코드 및 본 연구 코드를 최신 상태로 유지하기 위해 각각 다음 명령어로 업데이트(Pull)를 진행할 수 있습니다.

* **본 저장소 최신화 (DINO3)**:
  ```bash
  git pull origin main
  ```
* **의존성 저장소 최신화 (DINOv3)**:
  ```bash
  cd dinov3
  git pull origin main
  cd ..
  ```

### ③ 파이썬 필수 패키지 설치 (Python Package Installation)
본 저장소의 스크립트 실행과 노트북 시각화를 위해 필요한 파이썬 라이브러리 목록이 `requirements.txt`에 작성되어 있습니다. 아래 명령어로 필요한 패키지들을 일괄 설치할 수 있습니다.
```bash
pip install -r requirements.txt
```

---

## 1. 프로젝트 아키텍처 및 핵심 파일 구조

* **`train_disease_classifier.py`**: 캐시된 임베딩 벡터를 사용하여 Healthy/Diseased 이진 분류 헤드 모델을 학습하는 핵심 스크립트.
* **`train_crop_classifier.py`**: 데이터셋 로딩, 임베딩 캐싱 설정 및 다중 클래스/작물별 임베딩 학습 스크립트.
* **`analyze_global_distance.py`**: 전체 도메인 및 클래스 간의 Centroid 기반 Cosine Distance 행렬 계산 및 시각화 스크립트.
* **`analyze_crop_distance.py`**: 특정 작물(예: 사과, 토마토)을 필터링하여 Centroid 간의 거리 행렬을 계산 및 시각화하는 스크립트.
* **`extract_prediction_cases.py`**: 학습된 분류 모델을 평가하고, 도메인별 성공(TP, TN) 및 실패(FP, FN) 케이스를 분석하여 출력/저장하는 스크립트.
* **`plot_global_heatmaps.py`**: 계산된 Apple 및 Tomato Centroid Cosine Distance 히트맵을 한 번에 생성해주는 시각화 스크립트.
* **`plot_apple_distances.py`** / **`plot_tomato_distances.py`**: 각각 사과 및 토마토의 거리 시각화와 바 그래프 생성을 담당하는 보조 시각화 스크립트.

---

## 2. 베이스라인 (Baseline) 실험 개요

본 프로젝트에서는 DINOv3 임베딩 피처를 추출해 리니어 프로빙(Linear Probing)을 진행하는 제안 기법(Method)과 대비하기 위해 다음 베이스라인들이 실험되었습니다.
> [!NOTE]
> *본 레포지토리의 코드 단순화를 위한 정리 작업으로 이미지 기반 베이스라인 코드는 제거되었으나, 학습된 체크포인트 가중치는 `runs/` 폴더 내에 저장 및 평가되었습니다.*

1. **Random Initialization**: 임베딩 레이어를 무작위 초깃값에서부터 타겟 이진 라벨에 대해 스크래치(Scratch)로 학습.
2. **CNN Baseline**: ResNet-18 및 ResNet-50 구조를 타겟 질병 이진 라벨로 지도학습(Supervised).
3. **ImageNet Pretrained**: ImageNet으로 사전 학습된 가중치 백본을 활용해 파인튜닝(Fine-tuning) 및 선형 분석(Linear Probe) 진행.

---

## 3. 메서드 학습 방법 (Method Training)

DINOv3 모델을 통해 추출하여 캐싱해둔 임베딩 피처(`.pt` 파일)를 기반으로 빠르게 질병 분류 모델을 학습합니다.

```bash
# 기본 학습 명령어 (PlantVillage로 학습하고, PlantDoc과 PlantWild로 평가)
python3 train_disease_classifier.py \
    --train-domains PlantVillage \
    --eval-domains PlantDoc PlantWild \
    --max-samples 3000 \
    --seed 42
```

### 주요 파라미터 설명
* `--train-domains`: 학습 데이터로 활용할 도메인을 지정합니다 (기본값: `PlantVillage`).
* `--eval-domains`: 평가 및 일반화 성능을 검증할 도메인 목록입니다 (기본값: `PlantDoc`, `PlantWild`).
* `--max-samples`: 메모리 절약 및 균등 비교를 위해 도메인당 최대 샘플 수를 조절합니다 (기본값: `3000`, 0 지정 시 전체 사용).
* `--seed`: 무작위 샘플링 및 초기화를 고정하기 위한 시드 번호입니다.

---

## 4. UMAP / t-SNE 시각화 방법

고차원 임베딩 공간에서 도메인 간의 이동 경향(Domain Shift)과 질병 클래스의 군집 상태를 2차원에 투영하여 확인하기 위해 Jupyter Notebook 시각화 파일을 제공합니다.

### 실행 방법
1. Jupyter Lab 또는 VS Code를 활용하여 아래 노트북 파일을 실행합니다.
   * **`DomainCompare.ipynb`**: 임베딩 공간 내에서 특정 작물의 도메인 시프트(Domain Shift) 현상을 UMAP으로 투영.
   * **`DomainCompare_binary_healthy_disease.ipynb`**: UMAP과 t-SNE 분석 기법을 병행하여 도메인별/클래스별(Healthy/Disease) 분포 비교.
2. 노트북 내 정의된 `TARGET_CROP = "tomato"` 또는 `"apple"` 값을 변경하여 관심 있는 작물을 대상으로 차원 축소 그래프를 재생성할 수 있습니다.

---

## 5. Centroid Distance 계산 및 시각화 방법

두 군집의 분산과 배경 불일치에 왜곡되기 쉬운 Pairwise 평균 거리의 문제를 해결하기 위해, 각 군집의 **평균 위치(Centroid)를 구한 뒤 Cosine Distance**를 연산합니다. 자기 자신(동일 군집)과의 거리는 정확히 `0.0000`으로 맵핑됩니다.

### ① 전체 작물 통합 거리 계산
모든 작물을 통합하여 도메인 시프트 거리를 구하고, `runs/distance_analysis/`에 보고서와 Heatmap을 저장합니다.
```bash
python3 analyze_global_distance.py
```

### ② 특정 작물별 거리 계산
특정 작물(예: 토마토, 사과)에 대해 분석하려면 `--crop` 인자를 넘깁니다. 결과는 `runs/crop_distance_analysis/{crop_name}/` 하위에 저장됩니다.
```bash
# 토마토 분석
python3 analyze_crop_distance.py --crop tomato

# 사과 분석
python3 analyze_crop_distance.py --crop apple
```

### ③ 히트맵 직접 생성 및 플로팅
위 스크립트 실행으로 저장된 Centroid Distance 결과 수치를 모아 고해상도 히트맵을 생성합니다.
```bash
python3 plot_global_heatmaps.py
```
* **결과 이미지**: `apple_domain_distance.png`, `apple_class_distance.png`, `tomato_domain_distance.png`, `tomato_class_distance.png` 형태로 출력됩니다.

---

## 6. Failure Case 출력 방법

학습 완료된 분류 모델을 대상으로 실제로 오판한 사례(Failure Cases: False Positive, False Negative) 및 고확률 성공 사례(TP, TN)의 이미지 경로와 상세 확률을 추출합니다.

```bash
python3 extract_prediction_cases.py \
    --checkpoint runs/pv_to_pd_pw_binary_disease.pt \
    --eval-domains PlantDoc PlantWild \
    --output-dir runs/case_analysis \
    --top-k 10
```

### 주요 파라미터 설명
* `--checkpoint`: 평가에 사용할 학습 완료된 모델의 파일 경로를 지정합니다.
* `--top-k`: 각 성공/실패 케이스 유형별로 정렬하여 복사해 올 대표 이미지 개수입니다.
* `--output-dir`: 유형별 이미지 복사본 및 원본 정보가 담긴 CSV 메타데이터 파일이 저장되는 출력 디렉토리입니다.
