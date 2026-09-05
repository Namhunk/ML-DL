# 운수종사자 인지적 특성 데이터를 활용한 교통사고 위험 예측 AI 경진대회

### [Competition Link](https://dacon.io/competitions/official/236607/overview/description)

## 대회 개요

- **과제:** 운수종사자 자격검사(A: 신규자격, B: 자격유지) 과정에서 수집된 인지·반응 세부 검사 데이터를 활용해, 교통사고 위험군에 속할 확률을 예측
- **주최:** 행정안전부, 한국지능정보사회진흥원
- **주관:** 한국교통안전공단
- **운영:** Dacon
- **기간:** 2025.10.13 - 2025.11.14
- **성과:** 최종 **74등 / 437**

## 평가 지표

AUC뿐 아니라 **확률 보정(calibration)** 까지 함께 보는 복합 지표 — 낮을수록 좋음

```
score = 0.5 * (1 - AUC) + 0.25 * Brier + 0.25 * ECE
```

- **AUC:** 순위 판별력
- **Brier Score:** 예측 확률의 평균제곱오차
- **ECE (Expected Calibration Error):** 예측 확률과 실제 발생 빈도의 괴리

위험군을 잘 골라내는 것만으로는 부족하고 **확률값 자체가 신뢰할 수 있어야** 하는 문제\
Brier와 ECE가 합쳐서 50%를 차지한다는 점이 아래 모델 설계를 결정

## 데이터 구조

A 검사와 B 검사는 **서로 다른 컬럼 체계**를 가지므로 전 과정이 A/B 두 갈래로 분리

- **A 검사:** A1~A5 (5개 소검사) — 각 시행의 조건·정오·응답시간이 **쉼표로 이어붙인 문자열 시퀀스** 형태
- **B 검사:** B1~B8 (8개 소검사) — 동일하게 시퀀스 문자열

## 베이스라인

전처리 골격은 Dacon 공식 베이스라인(Train / Inference)을 그대로 사용

> 시퀀스 문자열 → 평균·표준편차·비율 축약(`seq_mean`, `seq_std`, `seq_rate`, `masked_mean_*`)\
> → A1~A5 / B1~B8 소검사별 기본 피처 → 조건 간 차이(gap)·속도-정확도 트레이드오프·`rt_cv`·`RiskScore` 파생\
> → 피처 정렬(`align_to_model`) → 단일 LightGBM 예측

베이스라인의 검증 AUC는 A 0.5108 / B 0.4999 로 사실상 무작위 수준\
아래는 이 골격 위에 직접 추가한 부분만 기록

## 접근 방법

### 1. 시퀀스 통계 함수 확장

베이스라인은 시퀀스를 **평균·표준편차·비율** 세 가지로만 축약\
반응시간 분포의 형태와 시행 간 변동을 담기 위해 함수를 추가

| 추가 함수 | 의미 |
| :--- | :--- |
| `seq_median` / `seq_min` / `seq_max` | 분포의 위치 |
| `seq_quantile` (Q1/Q3 → IQR) | 분포의 퍼짐 |
| `seq_skew` / `seq_kurt` | 왜도·첨도 (표준편차 0일 때 방어 처리 포함) |
| `seq_cv` | 변동계수 = 주의력 일관성 |
| `seq_diff_mean` / `seq_diff_std` | 연속 시행 간 변화량 → 반응 안정성 |
| `seq_rolling_std_mean` | 국소 변동성의 평균 → 시행 중 집중력 흔들림 |
| `masked_std_from_csv_series` | **조건별 표준편차** — 베이스라인에는 조건별 평균만 존재 |

평균만 보면 "느린 사람"만 잡히고, **분포가 흔들리는 사람**은 놓친다는 판단

### 2. 소검사 피처 세분화

**A3 — 3분류 → 4분면 세분화**

베이스라인은 valid / invalid / correct 세 비율만 계산\
`A3-5` 시퀀스를 네 경우로 완전히 분해

| 추가 피처 | 의미 |
| :--- | :--- |
| `A3_vc_ratio` / `A3_vi_ratio` / `A3_ic_ratio` / `A3_ii_ratio` | valid·invalid × correct·incorrect 4분면 비율 |
| `A3_valid_correct_rate` | valid 시행 안에서의 정답률 |
| `A3_invalid_correct_rate` | invalid 시행 안에서의 정답률 |

전체 정답률이 같아도 **어느 조건에서 틀렸는지**가 다르면 인지 특성이 다르다는 가정

**A4 (Stroop) — 대폭 확장**

베이스라인은 `stroop_diff` 등 6개\
정오·응답 조건별 반응시간의 평균과 **표준편차**, 그리고 분포 통계 전반을 추가

- `A4_correct_rt_mean` / `A4_incorrect_rt_mean`, `A4_resp0_rt_mean` / `A4_resp1_rt_mean`
- 위 네 조건 각각의 표준편차 (`*_rt_std`)
- `A4_rt_median` / `min` / `max` / `skew` / `kurt` / `diff_mean` / `diff_std` / `Q1` / `Q3` / `IQR` / `rolling_std_mean`

**A1 / A2 — 누락 조건 및 분포 보강**

- `A1_rt_normal`, `A2_rt_cond1_normal`, `A2_rt_cond2_normal` — 베이스라인이 slow/fast만 쓰고 건너뛴 **normal 조건**
- 각 소검사의 Q1 / Q3 / IQR

**공통 — 응시 이력**

```python
df = df.sort_values(by=['PrimaryKey', 'TestDate'])
df['past_attempts'] = df.groupby('PrimaryKey').cumcount()
```

동일인의 **과거 응시 횟수**. 재응시자는 검사에 익숙해져 점수가 좋아지므로, 이를 보정하지 않으면 실제 위험도를 과소평가

### 3. 2단계 스태킹 + Calibration

베이스라인은 단일 LightGBM의 `predict_proba` 를 그대로 제출\
지표의 절반이 Brier·ECE이므로 **확률 보정**을 위한 2단계 구조를 추가

```
[1단계] 3-Seed 앙상블 (CalibratedClassifierCV 로 감싼 부스팅 모델)
            ↓  predict_proba 평균 → OOF_Pred
[2단계] META 모델 (OOF_Pred 단일 피처를 입력받는 보정 모델)
            ↓
        최종 확률
```

- 1단계: 서로 다른 시드의 예측 확률을 `np.clip(p, 1e-7, 1-1e-7)` 로 잘라 평균 → AUC 안정화
- 2단계: 1단계 출력만 입력받는 메타 모델이 확률 스케일을 재조정 → ECE / Brier 개선
- A/B 각각 독립적으로 FINAL + META 두 쌍을 학습 (총 4개 모델 파일)

`align_to_model()` 도 단일 모델이 아닌 **보정된 앙상블 리스트**에서 피처 순서를 읽도록 수정

```python
base_hist_model = model_list[0].calibrated_classifiers_[0].estimator
feat_names = base_hist_model.feature_names_in_
```

## 파일 구성

```
교통사고 위험 예측 AI 대회/
├─ EDA.ipynb                # 탐색적 데이터 분석
├─ inference.py             # 전처리 + 파생 + 2단계 앙상블 추론 (전체 파이프라인 포함)
├─ model/
│  ├─ model_A_FINAL.pkl     # A 검사 1단계 (3-Seed 리스트)
│  ├─ model_A_META.pkl      # A 검사 2단계 보정 모델
│  ├─ model_B_FINAL.pkl     # B 검사 1단계
│  └─ model_B_META.pkl      # B 검사 2단계
└─ requirements.txt
```

## 실행 방법

```bash
pip install -r requirements.txt
python inference.py
```

기본 경로 (`inference.py` 상단 `main()` 에서 수정)

```
./data/test.csv, ./data/test/A.csv, ./data/test/B.csv, ./data/sample_submission.csv
./model/*.pkl
→ ./output/submission.csv
```

## 회고

**잘 된 점**

- 평가 지표의 절반이 calibration이라는 점을 파악하고, 베이스라인에 없던 2단계 보정 구조를 넣은 것
- 조건별 평균만 보던 베이스라인에 **분포 통계와 조건별 표준편차**를 더해 피처 축을 넓힌 것
- `past_attempts` 로 재응시 효과를 보정한 것

**개선할 점**

- **전처리 골격을 베이스라인 그대로 사용** — 시퀀스를 스칼라로 축약하는 방식 자체는 손대지 않음\
  시계열을 그대로 입력받는 1D CNN·GRU 같은 접근을 시도하지 못함
- 전처리 함수 상당수가 `progress_apply` 기반 행 단위 루프라 느림\
  `masked_mean_from_csv_series` 처럼 벡터화된 방식으로 통일할 여지
- `preprocess_A` / `preprocess_B` 에 A1~A5, B1~B8 블록이 거의 복붙 구조 (베이스라인 그대로)\
  소검사 스펙을 dict로 선언하고 루프로 생성하면 크게 축소 가능
- **B 검사는 베이스라인에서 거의 손대지 않음** — A에 적용한 분포 통계·조건 세분화를 B에도 확장 가능
