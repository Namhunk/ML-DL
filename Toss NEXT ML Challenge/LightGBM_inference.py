import pandas as pd
import polars as pl
import pandas as pd
import numpy as np
import joblib
import glob
import lightgbm as lgb
from tqdm import tqdm

# Test 데이터 로드
test = pl.read_parquet('./test.parquet').drop('ID')

# categorical, numerical 분류
target_col = 'clicked'
cols_to_drop = ['seq', 'clicked']
cat_cols = ['gender', 'age_group', 'inventory_id', 'day_of_week', 'hour']
features = [col for col in test.columns if col not in cols_to_drop]
numeric_cols = [col for col in features if col not in cat_cols]

test_data = test.to_pandas()[features].copy() # pandas로 변환
test_data[cat_cols] = test_data[cat_cols].astype('category') # category 타입으로 변환
test_data[numeric_cols] = test_data[numeric_cols].astype('float32')

# 저장된 모든 모델 파일 경로 가져오기
model_files = glob.glob('lgbm_models/*.joblib')
print(f"Found {len(model_files)} models to ensemble.")

all_predictions = []
for model_file in tqdm(model_files):
    # 모델 로드
    loaded_model = joblib.load(model_file)
    # 예측 수행
    preds = loaded_model.predict(test_data, num_iteration=loaded_model.best_iteration)
    all_predictions.append(preds)
    
# 모든 예측 결과의 평균 계산
final_predictions = np.mean(all_predictions, axis=0)

# 제출 파일 생성
sub = pd.read_csv('sample_submission.csv')
sub['clicked'] = final_predictions
sub.to_csv('LightGBM.csv', index=False)