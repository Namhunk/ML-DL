# Track1 알고리즘 부문 : K리그-서울시립대 공개 AI 경진대회

### [Competition Link](https://dacon.io/competitions/official/236647/overview/description)

## 대회 개요

- **과제:** K리그 경기 내 주어진 플레이 시퀀스의 마지막 패스 도착 좌표(end_x, end_y)를 예측
- **주최/주관:** 서울시립대, 한국프로축구연맹
- **후원:** 서울시립대
- **운영:** Dacon
- **기간:** 2025.12.01 - 2026.01.12
- **성과:** 최종 **5등 / 937**

## 평가 지표

예측 좌표와 실제 좌표 사이의 **평균 유클리드 거리(m)** — 낮을수록 좋음

```
score = mean( sqrt((pred_x - true_x)^2 + (pred_y - true_y)^2) )
```

경기장 규격은 105m x 68m 고정

## 접근 방법

### 1. 문제 재정의: 절대 좌표 → 상대 변위 예측

end_x, end_y를 직접 맞추는 대신 **마지막 이벤트 시작점 기준 변위(dx, dy)** 를 예측

- 좌표를 0~1로 정규화(x/105, y/68)한 뒤 dx, dy를 학습
- 추론 시 `last_start + pred_d` 로 복원 후 경기장 범위로 clip

### 2. 사전 통계 맵 구축 (학습 데이터 전체 기준)

| 맵 | 내용 |
| :--- | :--- |
| `player_map` | 선수별 dx/dy 평균·최대·최소·이동범위, start_x/y 중앙값 |
| `team_map` | 팀별 start_x/y 중앙값 |
| `grid_map` | 경기장을 5×5 grid로 나눈 뒤 선수 × 존별 dx/dy 평균 (홈/원정 공격 방향 정규화 적용) |
| `type_med` | 이벤트 타입별 이동 거리 중앙값 |
| `point` | 필드 106×69 격자점을 전수 탐색해 dx/dy와 상관계수가 가장 높은/낮은 좌표 4개 추출 |

신규 선수·팀은 전체 평균(`global`)으로 대체해 결측 방지

### 3. Feature Engineering

시퀀스를 `game_episode` 단위로 묶고 마지막 이벤트를 기준으로 약 300여 개의 피처를 생성

- **마지막/이전 이벤트 정보:** 좌표, 선수 ID, 팀 ID, 행동 타입, 홈 여부
- **연결 정보:** 이전 종료 위치와 마지막 시작 위치의 차이, 경과 시간으로 나눈 속도, 속도 × 변위, 절댓값 변위
- **통계 맵 조인:** 선수·팀·grid 통계, 마지막 선수와 이전 선수의 dy/dx 비율 곱
- **윈도우 피처(window=3):** 최근 3개 이벤트의 dx/dy 평균, 변동계수, quantile, 윈도우 평균과 직전 값의 차이, 벡터 내적, 이전 그룹의 start_x/start_y 상관계수
- **필드 기하 피처:** 상관관계 상위 4개 좌표까지의 거리, 벡터 내적·외적, 중앙까지의 거리와 그 속도, 골대 각도 및 가시 각도, 페널티 박스 진입 여부

### 4. 모델: AutoGluon MAE + Quantile 이중 구조

dx, dy 각각에 대해 **2종류의 TabularPredictor**를 `presets='best_quality'` 로 학습 (총 4개 모델)

| 모델 | eval_metric | 역할 |
| :--- | :--- | :--- |
| MAE 모델 | `mean_absolute_error` | 안정적인 중심 추정 |
| Quantile 모델 | `pinball_loss` (q=0.2 / 0.5 / 0.8) | 예측 분포의 꼬리 보정 |

dx와 dy는 서로 다른 피처 집합을 사용 (`drop_columns()` 로 타깃별 불필요 피처 제거)

### 5. 조건부 블렌딩 + Optuna 최적화

Quantile 모델의 중앙값(q=0.5) 예측이 특정 구간에 들어갈수록 하위/상위 quantile 쪽으로 선형적으로 끌어당기는 가중치를 적용

```python
def get_blend_weight(pred, lower, upper, max_weight):
    position = np.clip((pred - lower) / (upper - lower), 0, 1)
    return position * max_weight
```

- 중앙값이 작을 때 → q=0.20 쪽으로 (`xwl`, `ywl`)
- 중앙값이 클 때 → q=0.80 쪽으로 (`xwh`, `ywh`)
- 마지막으로 MAE 모델과 `ratio : (1-ratio)` 비율로 최종 블렌딩

**핵심:** 이 블렌딩의 13개 파라미터(구간 경계 6개 × 2축 + ratio)를 OOF 예측 위에서 **Optuna 3,000 trial** 로 직접 최적화

목적함수가 곧 대회 지표(평균 유클리드 거리)이므로 지표를 직접 최소화하는 구조

## 파일 구성

```
서울 시립대 경진대회/
├─ EDA.ipynb              # 탐색적 데이터 분석
├─ Autogluon_train.py     # 통계 맵 생성 → 피처 생성 → 4개 모델 학습 → Optuna 블렌딩 최적화
├─ inference.py           # 저장된 맵/파라미터/모델 로드 → 추론 → 제출 파일 생성
└─ requirements.txt        # 의존성
```

학습 시 생성되는 산출물

```
autogluon_models/model_{dx,dy}_{mae,quantile}/   # AutoGluon 모델
params/feature_maps.joblib                       # player/team/grid/point/type_med 맵
params/best_params.yaml                          # Optuna 최적 블렌딩 파라미터
```

## 실행 방법

데이터는 `./open_track1/` 에 위치 (`train.csv`, `test.csv`, `match_info.csv`, `sample_submission.csv`)

```bash
pip install -r requirements.txt
python Autogluon_train.py
python inference.py
```

결과: `Autogluon_with_quantile_mae.csv`

## 회고

**잘 된 점**

- 절대 좌표 대신 변위를 예측한 것
- 대회 지표를 목적함수로 삼아 후처리 블렌딩을 직접 최적화한 것 — 순위에 가장 크게 기여

**개선할 점**

- `drop_columns()` 의 제거 피처 목록이 수동 하드코딩이라 재현·유지보수가 어려움\
  permutation importance 기반 자동 선택으로 대체할 여지
- Optuna가 OOF 예측 위에서 3,000 trial을 도는 만큼 블렌딩 파라미터가 검증셋에 과적합될 위험\
  nested validation으로 확인 필요
- `create_feature()` 가 에피소드 단위 파이썬 루프라 데이터가 커지면 병목
