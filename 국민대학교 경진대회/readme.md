# 제3회 국민대학교 AI빅데이터 분석 경진대회 
### [Competition Link](https://dacon.io/competitions/official/236619/overview/description)

- 무역 품목 간 공행성 쌍 판별 및 후행 품목 무역량 예측 AI 모델 개발
- 원시 무역 수입 데이터(2022년 1월 ~ 2025년 7월)를 기반으로, 품목 간 공행성(comovement)이 존재하는 선후행 쌍을 예측하고, 공행성이 있다고 판단된 경우에는 후행 품목의 다음 달(2025년 8월) 총 무역량(value)을 예측하는 AI 모델을 개발
- 원시 무역 데이터를 분석하여 품목 간 선후행 관계가 존재하는 공행성 쌍(A → B)을 찾아야 하며, 이후에는 선행 품목(A)의 흐름을 활용해 후행 품목(B)의 다음달의 총 무역량(value)을 예측
- 주최: 국민대학교 경영대학원, 기계산업진흥회(KOAMI)
- 주관: 국민대학교 경영대학, 국민대학교 경영대학원 AI빅데이터전공/디지털마케팅전
- 운영: Dacon
- 성과: 최종(64등 / 960)

---

- **공행성쌍 탐색 파라미터:** max_lag = 12, min_nonzero = 36, corr_threshold=0.35
- LightGBM, XGBoost, Histgradientboosting 을 메인으로 사용
- 1개의 SEED, 2024-07 부터 2024-06 까지의 데이터를 사용해 12개의 fold로 학습
- Linear, Ridge, Lasso, MLP를 서브 모델로 사용
- 각 모델의 가중치를 다르게 하여 결과를 구함
