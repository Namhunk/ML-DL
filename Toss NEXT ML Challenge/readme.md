# Toss NEXT ML Challenge: CTR 예측 모델링

### [Competition Link](https://dacon.io/competitions/official/236575/overview/description)

- 광고 클릭 예측(CTR) 모델 개발
- 앱 내 광고 지면에서의 외부 광고 노출 및 클릭 데이터를 기반으로 사용자가 광고를 클릭할 확률을 예측
- 주최: Toss, Dacon

---
  ### 주요 특징
  - **클래스 불균형:** click, non-click의 데이터를 1 : 3의 비율로 다운샘플링 및 Scale_pos_weight 파라미터 설정
  - **3-Fold Ensemble:** 3개의 서로 다른 시드를 사용하여 9개의 모델 예측 확보
  - **Optuna 하이퍼 파라미터 튜닝:** Optuna를 사용하여 최적의 하이퍼 파라미터 탐색

---
