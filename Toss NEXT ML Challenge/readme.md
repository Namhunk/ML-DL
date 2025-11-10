# Toss NEXT ML Challenge: CTR 예측 모델링

### [Competition Link](https://dacon.io/competitions/official/236575/overview/description)

- 광고 클릭 예측(CTR) 모델 개발
- 앱 내 광고 지면에서의 외부 광고 노출 및 클릭 데이터를 기반으로 사용자가 광고를 클릭할 확률을 예측
- 주최: Toss, Dacon

---
  ### 주요 특징
  - **클래스 불균형:** clicked, non-clicked의 데이터를 1 : 3의 비율로 다운샘플링 및 Scale_pos_weight 파라미터 설정
  - **3-Fold Ensemble:** 3개의 서로 다른 시드를 사용하여 9개의 모델 예측 확보
  - **Optuna 하이퍼 파라미터 튜닝:** Optuna를 사용하여 최적의 하이퍼 파라미터 탐색
    
---
### 1. 데이터 전처리
- **다운샘플링:** clicked : non-clicked = 1 : 3
- **범주형 데이터 Category type 변환**

### 2. 모델 학습
- **단일 모델 사용:** LightGBM
- **Optuna 하이퍼 파라미터 튜닝:** 전체 데이터를 train, validation으로 구분, 고정하고 사용할 하이퍼 파라미터 탐색
- **3-Fold Stratified cross Validation With Seed Ensemble:**  3개의 다른 시드 사용, 각 시드당 3-Fold CV 수행
- **Early Stopping:** 300(과적합 방지)

### 3. 앙상블
- 각 모델의 예측 결과를 평균내어 예측값 생성

---
### 개선할 점
- 더 다양한 모델들과의 앙상블
