# Toss NEXT ML Challenge: CTR 예측 모델링

### [Competition Link](https://dacon.io/competitions/official/236575/overview/description)

## 대회 개요

- **과제:** 앱 내 광고 지면의 외부 광고 노출/클릭 데이터를 기반으로 사용자가 광고를 클릭할 확률(CTR)을 예측
- **주최:** Toss, Dacon
- **기간:** 2025.09.08 - 2025.10.13
- **성과:** 최종 **52등 / 709**

## 평가 지표

```
score = 0.5 * AP + 0.5 * (1 / (1 + WLL))
```

- **AP (Average Precision):** 극단적 불균형 상황에서 PR 곡선 하단 면적
- **WLL (Weighted LogLoss):** 클래스별 LogLoss를 0.5씩 균등 가중

```python
WLL = 0.5 * (-mean(log(1 - p[y=0]))) + 0.5 * (-mean(log(p[y=1])))
```

소수 클래스인 클릭 데이터에 절반의 가중치가 걸림\
**불균형 처리가 곧 점수**인 문제

## 데이터

- 형식: `train.parquet` / `test.parquet` (Polars로 로드)
- 타깃: `clicked` (0/1, 극심한 불균형)
- 범주형: `gender`, `age_group`, `inventory_id`, `day_of_week`, `hour`
- 제외 컬럼: `seq`, `clicked`
- 나머지는 모두 수치형 → `float32` 다운캐스팅으로 메모리 절감

## 접근 방법

### 1. 클래스 불균형 대응 (이중 전략)

| 전략 | 내용 |
| :--- | :--- |
| 다운샘플링 | `clicked=1` 전량 유지 + `clicked=0` 을 3배수만 샘플링 → **1 : 3** 비율 구성 |
| `scale_pos_weight` | 다운샘플링 후 잔여 불균형을 `neg_count / pos_count` 로 추가 보정 |

다운샘플링만으로 비율을 1:1까지 낮추지 않고 1:3에서 멈춘 뒤 가중치로 마무리한 구성\
데이터를 과도하게 버리지 않으면서 학습 시간을 줄이는 절충안

### 2. Optuna 하이퍼파라미터 탐색

- 전체 데이터를 stratified 8:2로 train/valid 분할해 **고정**한 뒤 탐색 (CV 대신 홀드아웃 → 탐색 비용 절감)
- 목적함수: valid AUC 최대화, 50 trials
- `MedianPruner(n_warmup_steps=10000)` + `LightGBMPruningCallback` 으로 가망 없는 trial 조기 중단

탐색된 최적 파라미터

```python
{
    'num_leaves': 190,
    'learning_rate': 0.0016691651006236685,
    'colsample_bytree': 0.686387931453528,
    'subsample': 0.6314368795644892,
    'reg_alpha': 3.113738376402282,
    'reg_lambda': 0.0654507771673197,
    'min_child_samples': 468,
}
```

학습률이 매우 낮고(`0.0017`) `min_child_samples`가 큰 편(468)\
early stopping 300과 맞물려 과적합을 강하게 억제하는 조합

### 3. Seed × Fold 앙상블

```
for seed in [42, 2024, 1004]:
    StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
        for fold in 3 folds:
            LightGBM 학습 (early_stopping=300)
            → lgbm_models/lgbm_seed_{seed}_fold_{fold}.joblib 저장
```

**3 seeds × 3 folds = 총 9개 모델**

시드마다 fold 분할 자체가 달라지므로 단순 K-Fold보다 다양성 확보\
OOF 예측은 시드 축으로 평균해 검증 점수를 산출

### 4. 추론

`lgbm_models/*.joblib` 를 전부 로드해 9개 예측을 **산술 평균**\
각 모델은 저장된 `best_iteration` 으로 예측

## 파일 구성

```
Toss NEXT ML Challenge/
├─ LightGBM_Train.ipynb      # 다운샘플링 → Optuna → Seed×Fold 학습 → OOF 검증
├─ LightGBM_inference.py     # 9개 모델 로드 → 평균 → 제출 파일 생성
└─ requirements.txt
```

## 실행 방법

```bash
pip install -r requirements.txt
```

1. `train.parquet` 을 작업 디렉터리에 두고 `LightGBM_Train.ipynb` 실행 → `lgbm_models/` 에 9개 모델 저장
2. `test.parquet`, `sample_submission.csv` 를 두고 추론 실행

```bash
python LightGBM_inference.py
```

결과: `LightGBM.csv`

## 회고

**잘 된 점**

- 지표가 AP와 WLL의 결합이라는 점에서 불균형 처리를 최우선으로 잡은 것
- 다운샘플링과 `scale_pos_weight`를 함께 쓴 이중 전략

**개선할 점**

- **단일 모델(LightGBM)만 사용** — CatBoost, XGBoost 등과의 이종 앙상블 여지가 큼
- **피처 엔지니어링이 사실상 부재** — 원본 컬럼을 타입 변환만 해서 그대로 투입\
  유저×광고 상호작용, 시간대별 CTR 통계, target encoding 등이 CTR 문제에서 통상 큰 폭의 개선
- `seq` 컬럼을 그냥 드롭 — 시퀀스 정보라면 집계 피처로 활용 가능
- Optuna를 홀드아웃 1분할로만 실행해 파라미터가 그 분할에 편향됐을 가능성
