# PUBG Analysis
### 🔗[기존 README 파일 클릭 시 이동](./origin_README.md)
PUBG(PlayerUnknown's Battlegrounds) 데이터 분석 프로젝트 (수정 진행 중 입니다.)

---

## 프로젝트 개요

### 🎯 목표
배틀그라운드에서 **게임 순위를 높이는 전략**을 도출하는 것이 이 프로젝트의 핵심입니다.  
특정 맵과 클러스터에서 얼마만큼의 **이동거리와 데미지**를 내야 우승 확률이 높은지를 분석합니다.

### 📊 데이터
- **사용 기간**: 2017~2018년 PUBG 매치 데이터
- **분석 범위**: 주요 맵인 **에란겔(Erangel)** 과 **미라마(Miramar)**
- **주요 변수**:  
  - 이동 거리(`player_dist_total`, `player_dist_drive`)  
  - 데미지(`player_dmg`)  
  - 파티 크기 및 스쿼드 유형(솔로/듀오/스쿼드)  
  - 생존 시간(`player_survive_time`)  
  - 킬 수(`player_kills`)  

### 🧩 전제 조건
- 플레이어는 **3가지 유형**으로 나뉜다고 가정:
  1. **교전 선호형**: 적극적으로 교전을 즐기는 플레이어
  2. **생존 우선형(존버)**: 교전을 최소화하고 생존을 우선하는 플레이어
  3. **균형형**: 적당히 파밍하고 교전을 선택적으로 즐기는 일반적인 플레이어
- **맵별**로 교전 지역 및 플레이 스타일이 다르다는 점을 고려하여 분석.

---

## 분석 내용

### 1️⃣ 클러스터링을 통한 플레이어 유형 분류
- **사용 변수**:
  - `player_dist_total` (총 이동 거리)
  - `player_dmg` (가한 데미지)
  - `drive_type` (차량 이동 여부)
  - `player_kills` (킬 수)
- **방법**:
  - 데이터 표준화: `StandardScaler()`
  - 클러스터링: `KMeans()`로 3개의 클러스터로 분류
- **결과**:
  - **Cluster 0**: 일반적 플레이어
  - **Cluster 1**: 교전 선호형 플레이어
  - **Cluster 2**: 생존 우선형 플레이어 (존버)

### 2️⃣ 생존 확률 분석
- 각 클러스터의 생존 확률을 비교하여 유형별 전략적 우위를 분석.
- **`player_dist_drive`(차량 이동 거리)** 가 생존 확률에 큰 영향을 미치는 것으로 나타남.  
  이는 특정 상황에서 차량 이동 전략이 중요하다는 점을 시사합니다.

---

## 모델링

### 🔍 사용 모델
- 다양한 머신러닝 모델을 활용하여 생존 확률 예측:
  - **Decision Tree**
  - **Random Forest**
  - **Logistic Regression**
  - **Gradient Boosting**

### 성능 평가
- **Random Forest**와 **Decision Tree**가 높은 정확도를 보였으나,  
  **속도**를 고려하여 **Decision Tree**를 최종 모델로 채택.

### 최종 모델 정의
- 주요 변수:
  - `player_dist_total`, `player_dmg` (이동 거리와 데미지)
  - 각 클러스터 원-핫 인코딩 (`cluster_0`, `cluster_1`, `cluster_2`)
- 타겟 변수:
  - **상위 5위 이내 진입 여부** (이진 분류)

```python
# 예측 모델 데이터 준비
X = sample_df[['player_dist_total', 'player_dmg', 'cluster_0', 'cluster_1', 'cluster_2']]
y = (sample_df['team_placement'] <= 5).astype(int)  # 상위 5위 여부
```

---

## 결론 및 기대 효과

1. **유형별 플레이 전략**:
   - 유형에 따라 최적의 생존 전략 및 우승 확률을 높이는 방법 제안.
   - 예: 차량 활용, 특정 지역 우선 파밍 등.

2. **맵별 최적화 전략**:
   - 에란겔과 미라마의 특성에 맞춘 데이터 기반 플레이 가이드 제공.

3. **게임 데이터 활용 가능성**:
   - e스포츠 전략 분석, 신규 사용자 유입을 위한 데이터 기반 인사이트 제공.

---

## 프로젝트 구조

```plaintext
PUBG_analysis/
├── data/                           # 전처리된 데이터 (원본 데이터 용량 이슈)
├── notebooks/                      # 분석 Jupyter 노트북
├── models/                         # 추가 예정: 훈련된 모델 저장
├── results/                        # 추가 예정: 결과 데이터 및 시각화 자료
├── src/                            # 수정 중..
│   ├── clustering/                 
│   │     └── kmeans.py             # k-means로 클러스터링
│   ├── data/
│   │     ├── download.py           # kaggle dataset download
│   │     └── preprocess.py         # 전처리 진행
│   ├── modeling/
│   │     ├── compare_models.py     
│   │     ├── evaluate.py            
│   │     ├── predict.py             
│   │     └── train.py   
│   ├── serving/
│   │     └── api.py  
│   └── visualization/
│         └── plot.py  
├── origin_README.md                # 기존 설명 자료 (시각화 포함)
└── README.md                       # 프로젝트 설명
```

---

## 주요 기술 스택
- **Python**: 데이터 분석 및 모델링
- **Pandas, NumPy**: 데이터 처리
- **Scikit-learn**: 머신러닝 모델링 및 평가
- **Matplotlib, Seaborn**: 데이터 시각화

---

## 참고 자료
- 데이터 출처: Kaggle
- 분석 환경: Python 3.9, Jupyter Notebook

---
