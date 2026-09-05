# 제3회 국민대학교 AI빅데이터 분석 경진대회

### [Competition Link](https://dacon.io/competitions/official/236619/overview/description)

## 대회 개요

- **과제:** 원시 무역 수입 데이터(2022.01 ~ 2025.07)를 기반으로
  1. 품목 간 **공행성(comovement)이 존재하는 선후행 쌍 (A → B)** 을 판별
  2. 공행성이 있다고 판단된 쌍의 **후행 품목 B의 다음 달(2025.08) 총 무역량(value)** 을 예측
- **주최:** 국민대학교 경영대학원, 기계산업진흥회(KOAMI)
- **주관:** 국민대학교 경영대학, 국민대학교 경영대학원 AI빅데이터전공/디지털마케팅전공
- **운영:** Dacon
- **기간:** 2025.11.10 - 2025.11.28
- **성과:** 최종 **64등 / 960**

## 평가 지표

두 과제를 6:4로 결합한 복합 지표 — 높을수록 좋음

```
score = 0.6 * S1 + 0.4 * S2

S1 = 공행성쌍 F1 (pair 집합 기준 precision/recall)
S2 = 1 - NMAE
```

NMAE는 정답 집합 G와 제출 집합 P의 **합집합 U = G ∪ P** 위에서 계산

- 양쪽 모두에 있는 쌍: `min(|y-ŷ| / (|y|+eps), 1.0)`
- FP 또는 FN인 쌍: **오차 100% 로 간주**

**쌍을 잘못 찾으면 회귀 점수까지 같이 깎이는 구조**\
1단계 쌍 탐색이 전체 점수를 좌우

## 베이스라인

파이프라인 골격은 Dacon 공식 베이스라인을 그대로 사용

> 월별 피벗 생성 → lag correlation 기반 공행성쌍 탐색 → 쌍 × 시점별 학습 데이터 생성 → 회귀 → 음수 클리핑·정수 반올림 후 제출

아래는 이 골격 위에 직접 추가한 부분만 기록

## 접근 방법

### 1. 탐색 파라미터 재설정

베이스라인 기본값이 쌍을 과다 채택하는 경향이 있어 조정

| 파라미터 | baseline | 사용값 | 의도 |
| :--- | :---: | :---: | :--- |
| `max_lag` | 6 | **12** | 연 단위 주기를 갖는 무역 품목까지 포착 |
| `min_nonzero` | 12 | **36** | 43개월 중 36개월 이상 거래된 품목만 후보 |
| `corr_threshold` | 0.4 | **0.35** | 임계값은 낮추되 위 두 필터로 정밀도 확보 |

거래가 드문 품목은 우연히 높은 상관을 만들기 쉬움\
`min_nonzero` 를 12 → 36으로 크게 올려 F1의 precision을 방어

### 2. 피처 확장

베이스라인은 value 계열 5개(`b_t`, `b_t_1`, `a_t_lag`, `max_corr`, `best_lag`)만 사용\
여기에 무게(weight) 축과 품목 분류 정보를 추가

| 추가 피처 | 의미 |
| :--- | :--- |
| `b_w_t`, `b_w_t_1` | 후행 품목의 t, t-1 시점 **weight** |
| `a_w_t_lag` | 선행 품목의 t-lag 시점 **weight** |
| `a_hs2`, `b_hs2`, `a_hs4`, `b_hs4` | HS 코드 2자리/4자리 품목 분류 (category 타입) |
| `time` | 시점 인덱스 — 시간 가중치 및 fold 분할의 기준 |

금액(value)만 보면 단가 변동과 물량 변동이 섞임\
weight를 함께 넣어 두 축을 모델이 분리할 수 있게 함

### 3. 시계열 검증: Expanding Window 12-Fold

베이스라인에는 검증 절차가 없어 직접 구성\
일반 K-Fold는 미래 정보 누수가 발생하므로 **시점 기반 확장 윈도우**를 사용

```
fold k:  train = {time < vtime},  valid = {time == vtime}
         vtime = 2024-07 ... 2025-06  (총 12 fold)
```

추가로 **최근 데이터에 지수 가중치**를 부여

```python
norm_time = (t - t_min) / (t_max - t_min)
sample_weight = np.exp(norm_time * 3)
```

가장 오래된 시점 대비 최신 시점의 가중치가 약 20배(`e^3`)\
구조가 변하는 시계열에서 최근 패턴을 우선 학습시키려는 의도

검증 지표는 대회의 S2와 동일한 clipped relative error를 그대로 사용

### 4. 모델: LinearRegression → 7종 가중 앙상블

베이스라인의 단일 `LinearRegression` 을 다음 구성으로 교체

**메인 (트리 계열)** — 각각 Optuna 20 trials, SEED 1개로 튜닝

- LightGBM
- XGBoost (`enable_categorical=True`, `tree_method='hist'`)
- HistGradientBoosting (`categorical_features` 마스크 지정)

**서브 (선형/신경망)** — `StandardScaler` + `OneHotEncoder` 파이프라인

- LinearRegression / Ridge(α=1.0) / Lasso(α=0.01) / MLP(64, 32)

`FinalHybridModel` 클래스로 묶어 가중 투표하고, 그 위에 잔차 학습 모델을 추가

**최종 가중치**

```python
{'lgb': 0.5, 'xgb': 0.5, 'hgb': 4.0,
 'linear': 0.5, 'ridge': 0.5, 'lasso': 0.5, 'mlp': 3.5}
```

검증 결과에 따라 **HistGradientBoosting(4.0)과 MLP(3.5)에 압도적 비중**\
튜닝에 가장 공을 들인 LGBM/XGB는 오히려 0.5로 축소

## 파일 구성

```
국민대학교 경진대회/
├─ EDA.ipynb      # 탐색적 데이터 분석
└─ FInal.ipynb    # 피벗 생성 → 공행성쌍 탐색 → 학습 데이터 생성 → Optuna → 앙상블 → 제출
```

## 실행 방법

`train.csv` 를 노트북과 같은 디렉터리에 두고 `FInal.ipynb` 를 위에서부터 실행

결과: `final_submission.csv`

```
필요 패키지: pandas, numpy, scikit-learn, lightgbm, xgboost, catboost, optuna, tqdm
```

## 회고

**잘 된 점**

- 평가 지표에서 FP/FN이 회귀 점수까지 100% 오차로 반영된다는 구조를 파악하고 탐색 파라미터를 보수적으로 재설정한 것
- 베이스라인에 없던 시점 기반 확장 윈도우 검증을 구성해 누수를 차단한 것
- value 단일 축에 weight와 HS 분류를 더해 피처를 확장한 것

**개선할 점**

- **베이스라인 골격에 대한 의존이 큼**\
  공행성쌍 탐색 로직 자체는 손대지 않고 파라미터만 조정 — 상관계수 외에 Granger 인과성, DTW, 공적분 등 다른 판별 방식을 시도하지 못함
- 공행성쌍 탐색이 O(품목수² × lag) 이중 파이썬 루프 (베이스라인 그대로)\
  행렬 연산이나 FFT 기반 교차상관으로 대체 가능
- `max_lag`, `min_nonzero`, `corr_threshold` 세 값이 점수의 60%를 좌우하는데 수동 지정\
  F1을 목적함수로 튜닝했다면 개선 폭이 컸을 것
- 앙상블 가중치도 수동 지정 — Optuna로 최적화 가능
- `build_training_data()` 안에서 쌍마다 `df.loc[df['item_id'] == leader, 'hs4']` 로 원본을 스캔\
  HS 코드를 사전에 dict로 만들어두면 대폭 개선
- CatBoost를 import하지만 실제로는 미사용
