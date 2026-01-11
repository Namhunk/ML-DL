#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, argparse, joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
import lightgbm as lgb
from scipy.stats import skew, kurtosis
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
tqdm.pandas()


# 점수 계산 함수
def expected_calibration_error(y_true, y_prob, n_bins=10):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='uniform')
    bin_totals = np.histogram(y_prob, bins=np.linspace(0, 1, n_bins + 1), density=False)[0]
    non_empty_bins = bin_totals > 0
    bin_weights = bin_totals / len(y_prob)
    bin_weights = bin_weights[non_empty_bins]
    prob_true = prob_true[:len(bin_weights)]
    prob_pred = prob_pred[:len(bin_weights)]
    ece = np.sum(bin_weights * np.abs(prob_true - prob_pred))
    return ece
    
def auc_brier_ece(answer_df, submission_df):
    # Check for missing values in submission_df
    if submission_df.isnull().values.any():
        raise ValueError("The submission dataframe contains missing values.")


    # Check if the number and names of columns are the same in both dataframes
    if len(answer_df.columns) != len(submission_df.columns) or not all(answer_df.columns == submission_df.columns):
        raise ValueError("The columns of the answer and submission dataframes do not match.")
        
    submission_df = submission_df[submission_df.iloc[:, 0].isin(answer_df.iloc[:, 0])]
    submission_df.index = range(submission_df.shape[0])
    
    # Calculate AUC for each class
    auc_scores = []
    for column in answer_df.columns[1:]:
        y_true = answer_df[column]
        y_scores = submission_df[column]
        auc = roc_auc_score(y_true, y_scores)
        auc_scores.append(auc)


    # Calculate mean AUC
    mean_auc = np.mean(auc_scores)


    brier_scores = []
    ece_scores = []
    
    # Calculate Brier Score and ECE for each class
    for column in answer_df.columns[1:]:
        y_true = answer_df[column].values
        y_prob = submission_df[column].values
        
        # Brier Score
        brier = mean_squared_error(y_true, y_prob)
        brier_scores.append(brier)
        
        # ECE
        ece = expected_calibration_error(y_true, y_prob)
        ece_scores.append(ece)
    
    # Calculate mean Brier Score and mean ECE
    mean_brier = np.mean(brier_scores)
    mean_ece = np.mean(ece_scores)
    
    # Calculate combined score
    combined_score = 0.5 * (1 - mean_auc) + 0.25 * mean_brier + 0.25 * mean_ece
    
    return combined_score

# 각 시퀀스 통계
# 각 시퀀스 통계
def convert_age(val): # 나이 변환 
    if pd.isna(val): return np.nan
    try:
        base = int(str(val)[:-1])
        return base if str(val)[-1] == "a" else base + 5
    except:
        return np.nan

def split_testdate(val): # 날짜 변환
    try:
        v = int(val)
        return v // 100, v % 100
    except:
        return np.nan, np.nan

def seq_mean(series): # 각 시퀀스 평균
    return series.fillna("").progress_apply(
        lambda x: np.fromstring(x, sep=",").mean() if x else np.nan
    )

def seq_std(series): # 각 시퀀스 표준편차
    return series.fillna("").progress_apply(
        lambda x: np.fromstring(x, sep=",").std() if x else np.nan
    )

def seq_rate(series, target="1"): # 각 시퀀스에서 target이 차지하는 비율
    return series.fillna("").progress_apply(
        lambda x: str(x).split(",").count(target) / len(x.split(",")) if x else np.nan
    )

def masked_mean_from_csv_series(cond_series, val_series, mask_val): # 평균 반응 시간
    cond_df = cond_series.fillna("").str.split(",", expand=True).replace("", np.nan)
    val_df  = val_series.fillna("").str.split(",", expand=True).replace("", np.nan)

    cond_arr = cond_df.to_numpy(dtype=float)
    val_arr  = val_df.to_numpy(dtype=float)

    mask = (cond_arr == mask_val)
    with np.errstate(invalid="ignore"):
        sums = np.nansum(np.where(mask, val_arr, np.nan), axis=1)
        counts = np.sum(mask, axis=1)
        out = sums / np.where(counts==0, np.nan, counts)
    return pd.Series(out, index=cond_series.index)

def masked_mean_in_set_series(cond_series, val_series, mask_set): # 특정 집합의 평균
    cond_df = cond_series.fillna("").str.split(",", expand=True).replace("", np.nan)
    val_df  = val_series.fillna("").str.split(",", expand=True).replace("", np.nan)

    cond_arr = cond_df.to_numpy(dtype=float)
    val_arr  = val_df.to_numpy(dtype=float)

    mask = np.isin(cond_arr, list(mask_set))
    with np.errstate(invalid="ignore"):
        sums = np.nansum(np.where(mask, val_arr, np.nan), axis=1)
        counts = np.sum(mask, axis=1)
        out = sums / np.where(counts == 0, np.nan, counts)
    return pd.Series(out, index=cond_series.index)

# ---- 추가 ----
def seq_median(series): # 각 시퀀스 중앙값
    return series.fillna("").progress_apply(
        lambda x: np.median(np.fromstring(x, sep=",")) if x else np.nan
    )

def seq_min(series): # 각 시퀀스 최솟값
    return series.fillna("").progress_apply(
        lambda x: np.fromstring(x, sep=",").min() if x else np.nan
    )

def seq_max(series): # 각 시퀀스 최댓값
    return series.fillna("").progress_apply(
        lambda x: np.fromstring(x, sep=",").max() if x else np.nan
    )

def safe_skew(arr):
    # 데이터가 2개 미만이거나, 표준편차가 0에 매우 가까우면 계산하지 않음
    if len(arr) < 2 or np.std(arr) < 1e-9:
        return 0.0  # (또는 np.nan)
    return skew(arr)

def seq_skew(series): # 각 시퀀스 왜도
    return series.fillna("").progress_apply(
        lambda x: safe_skew(np.fromstring(x, sep=",")) if x else np.nan
    )

def safe_kurt(arr):
    if len(arr) < 2 or np.std(arr) < 1e-9:
        return 0.0
    return kurtosis(arr)

def seq_kurt(series): # 각 시퀀스 첨도
    return series.fillna("").progress_apply(
        lambda x: safe_kurt(np.fromstring(x, sep=",")) if x else np.nan
    )

def seq_diff_mean(series): # 연속 값 차이의 평균
    def calc_diff_mean(x):
        if not x: return np.nan
        arr = np.fromstring(x, sep=",")
        if len(arr) < 2: return np.nan
        return np.mean(np.diff(arr))
    return series.fillna("").progress_apply(calc_diff_mean)

def seq_quantile(series, q=0.75): # 각 시퀀스 특정 분위수 (기본값: 75%)
    return series.fillna("").progress_apply(
        lambda x: np.quantile(np.fromstring(x, sep=","), q) if x else np.nan
    )

def seq_diff_std(series): # 연속 값 차이의 표준편차
    def calc_diff_std(x):
        if not x: return np.nan
        try: # np.fromstring 에러 방지
            arr = np.fromstring(x, sep=",")
        except ValueError:
             return np.nan # 변환 실패 시 NaN
        if len(arr) < 2: return np.nan # 차이를 계산하려면 최소 2개 필요
        # np.diff 계산 결과가 하나뿐이면 std는 0 또는 NaN (ddof=1 기본값)
        diffs = np.diff(arr)
        if len(diffs) < 2: return 0.0 # 차이값이 하나면 변동성은 0
        return np.std(diffs, ddof=1)
    return series.fillna("").progress_apply(calc_diff_std)

def seq_rolling_std_mean(series, window_size=2):
    def calc_rolling_mean(x):
        if not x: return np.nan
        try:
            arr = np.fromstring(x, sep=",")
        except ValueError:
            return np.nan
        if len(arr) < window_size: return np.nan # 윈도우 크기보다 작으면 계산 불가

        s = pd.Series(arr)
        rolling_means = s.rolling(window=window_size, min_periods=1).std() # min_periods=1은 초반 NaN 방지
        return rolling_means.mean() # 롤링 표준편들의 평균
    
    return series.fillna("").progress_apply(calc_rolling_mean)

def calculate_cv(arr, ddof=1): # 변동 계수 계산
    mean = arr.mean()
    std = arr.std(ddof=ddof)
    
    if mean == 0 or np.isnan(mean):
        return np.nan
    
    return std / mean

def seq_cv(series, ddof=1): # 각 시퀀스 변동계수
    return series.fillna("").progress_apply(
        lambda x: calculate_cv(np.fromstring(x, sep=","), ddof=ddof) if x else np.nan
    )

def masked_std_from_csv_series(cond_series, val_series, mask_val): # 표준편차 반응 시간
    cond_df = cond_series.fillna("").str.split(",", expand=True).replace("", np.nan)
    val_df  = val_series.fillna("").str.split(",", expand=True).replace("", np.nan)
    cond_arr = cond_df.to_numpy(dtype=float)
    val_arr  = val_df.to_numpy(dtype=float)
    mask = (cond_arr == mask_val)
    masked_vals = np.where(mask, val_arr, np.nan)
    counts = np.sum(~np.isnan(masked_vals), axis=1)
    out = np.full(masked_vals.shape[0], np.nan)
    valid_rows_mask = (counts >= 2)
    
    if np.any(valid_rows_mask):
        std_results = np.nanstd(masked_vals[valid_rows_mask], axis=1, ddof=1)
        out[valid_rows_mask] = std_results
        
    return pd.Series(out, index=cond_series.index)

# ---- 추가 ----
# Train_A Feature Engineering
def preprocess_A(train_A):
    df = train_A.copy()
    original_index = df.index
    # ---- Age, TestDate ----
    print('Age, TestDate, PrimaryKey 파생')
    df['Age_num'] = df['Age'].map(convert_age) # Age: a -> 0, b -> 5 로 분류
    ym = df['TestDate'].map(split_testdate) # TestDate: Year, Month 로 분할
    df['Year'] = [y for y, m in ym]
    df['Month'] = [m for y, m in ym]

    df = df.sort_values(by=['PrimaryKey', 'TestDate']) # Key, Date 순 정렬
    df['past_attempts'] = df.groupby('PrimaryKey').cumcount() # 과거 횟수
    df = df.reindex(original_index)
    feats_list = []

    # ---- A1 ----
    print("A1 feature 생성")
    A1_resp_rate = seq_rate(df["A1-3"], "1").rename('A1_resp_rate') # 응답 비율
    A1_rt_mean   = seq_mean(df["A1-4"]).rename('A1_rt_mean') # 응답시간 평균
    A1_rt_std    = seq_std(df["A1-4"]).rename('A1_rt_std') # 응답시간 표준편차
    A1_rt_left   = masked_mean_from_csv_series(df["A1-1"], df["A1-4"], 1).rename('A1_rt_left') # 왼쪽 진행방향 응답시간 평균
    A1_rt_right  = masked_mean_from_csv_series(df["A1-1"], df["A1-4"], 2).rename('A1_rt_right') # 오른쪽 진행방향 응답시간 평균
    A1_rt_side_diff = (A1_rt_left - A1_rt_right).rename('A1_rt_side_diff') # 왼쪽-오른쪽 응답시간 차이
    A1_rt_slow   = masked_mean_from_csv_series(df["A1-2"], df["A1-4"], 1).rename('A1_rt_slow') # slow 응답시간 평균
    A1_rt_fast   = masked_mean_from_csv_series(df["A1-2"], df["A1-4"], 3).rename('A1_rt_fast') # fast 응답시간 평균
    A1_rt_speed_diff = (A1_rt_slow - A1_rt_fast).rename('A1_rt_speed_diff') # slow-fast 응답시간 차이
    # ---- 추가 ----
    A1_rt_normal = masked_mean_from_csv_series(df['A1-2'], df['A1-4'], 2).rename('A1_rt_normal') # normal 응답시간 평균
    A1_Q3        = seq_quantile(df['A1-4'], q=0.75).rename('A1_Q3') # 75% 분위수
    A1_Q1        = seq_quantile(df['A1-4'], q=0.25).rename('A1_Q1') # 25% 분위수
    A1_IQR       = (A1_Q3 - A1_Q1).rename('A1_IQR') # IQR

    A1_feats = [
        A1_resp_rate, A1_rt_mean, A1_rt_std, A1_rt_left,
        A1_rt_right, A1_rt_side_diff, A1_rt_slow, A1_rt_fast,
        A1_rt_speed_diff, A1_rt_normal,
        A1_Q3, A1_Q1, A1_IQR
    ]
    print(f'A1 feats count: {len(A1_feats)}') # 25
    feats_list.extend(A1_feats)

    # ---- A2 ----
    print('A2 feature 생성')
    A2_resp_rate = seq_rate(df["A2-3"], "1").rename('A2_resp_rate') # 응답 비율
    A2_rt_mean   = seq_mean(df["A2-4"]).rename('A2_rt_mean') # 응답시간 평균
    A2_rt_std    = seq_std(df["A2-4"]).rename('A2_rt_std') # 응답시간 표준편차
    A2_rt_cond1_slow = masked_mean_from_csv_series(df['A2-1'], df['A2-4'], 1).rename('A2_rt_cond1_slow') # cond1의 slow 응답시간 평균
    A2_rt_cond1_fast = masked_mean_from_csv_series(df['A2-1'], df['A2-4'], 3).rename('A2_rt_cond1_fast') # con1의 fast 응답시간 평균
    A2_rt_cond1_diff = (A2_rt_cond1_slow - A2_rt_cond1_fast).rename('A2_rt_cond1_diff') # slow-fast 응답시간 차이
    A2_rt_cond2_slow = masked_mean_from_csv_series(df['A2-2'], df['A2-4'], 1).rename('A2_rt_cond2_slow') # cond2의 slow 응답시간 평균
    A2_rt_cond2_fast = masked_mean_from_csv_series(df['A2-2'], df['A2-4'], 3).rename('A2_rt_cond2_fast') # cond2의 fast 응답시간 평균
    A2_rt_cond2_diff = (A2_rt_cond2_slow - A2_rt_cond2_fast).rename('A2_rt_cond2_diff')
    # ---- 추가 ----
    A2_rt_cond1_normal  = masked_mean_from_csv_series(df['A2-1'], df['A2-4'], 2).rename('A2_rt_cond1_normal') # cond1의 normal 응답시간 평균
    A2_rt_cond2_normal  = masked_mean_from_csv_series(df['A2-2'], df['A2-4'], 2).rename('A2_rt_cond2_normal') # cond2의 normal 응답시간 평균
    A2_Q3            = seq_quantile(df['A2-4'], q=0.75).rename('A2_Q3') # 75% 분위수
    A2_Q1            = seq_quantile(df['A2-4'], q=0.25).rename('A2_Q1') # 25% 분위수
    A2_IQR           = (A2_Q3 - A2_Q1).rename('A2_IQR') # IQR

    A2_feats = [
        A2_resp_rate, A2_rt_mean, A2_rt_std, A2_rt_cond1_slow,
        A2_rt_cond1_fast, A2_rt_cond1_diff, A2_rt_cond2_slow, 
        A2_rt_cond2_fast, A2_rt_cond2_diff, A2_rt_cond1_normal,
        A2_rt_cond2_normal, A2_Q3, A2_Q1, A2_IQR
    ]

    print(f'A2 feats count: {len(A2_feats)}') # 26

    feats_list.extend(A2_feats)

    # ---- A3 ----
    print('A3 feature 생성')
    s = df['A3-5'].fillna('') # (valid, invalid, correct, incorrect)
    total = s.apply(lambda x: len(x.split(',')) if x else 0)
    # ---- 수정---- A3 4가지 경우로 분류
    vc = s.apply(lambda x: sum(v == '1' for v in x.split(',')) if x else 0) # valid correct
    vi = s.apply(lambda x: sum(v == '2' for v in x.split(',')) if x else 0) # valid incorrect
    ic = s.apply(lambda x: sum(v == '3' for v in x.split(',')) if x else 0) # invalid correct
    ii = s.apply(lambda x: sum(v == '4' for v in x.split(',')) if x else 0) # invalid incorrect
    # 각 비율
    A3_vc_ratio = (vc/total).replace([np.inf,-np.inf], np.nan).rename('A3_vc_ratio')
    A3_vi_ratio = (vi/total).replace([np.inf,-np.inf], np.nan).rename('A3_vi_ratio')
    A3_ic_ratio = (ic/total).replace([np.inf,-np.inf], np.nan).rename('A3_ic_ratio')
    A3_ii_ratio = (ii/total).replace([np.inf,-np.inf], np.nan).rename('A3_ii_ratio')

    A3_resp2_rate = seq_rate(df['A3-6'], '1').rename('A3_resp2_rate') # y 응답 비율
    A3_rt_mean = seq_mean(df['A3-7']).rename('A3_rt_mean') # 응답시간 평균
    A3_rt_std = seq_std(df['A3-7']).rename('A3_rt_std') # 응답시간 표준편차
    A3_rt_size_small = masked_mean_from_csv_series(df["A3-1"], df["A3-7"], 1).rename('A3_rt_size_small')  # small일때의 rt 평균
    A3_rt_size_big   = masked_mean_from_csv_series(df["A3-1"], df["A3-7"], 2).rename('A3_rt_size_big') # big일때의 rt 평균
    A3_rt_size_diff  = (A3_rt_size_small - A3_rt_size_big).rename('A3_rt_size_diff') # 크기에 따른 평균 시간의 차이
    A3_rt_side_left  = masked_mean_from_csv_series(df["A3-3"], df["A3-7"], 1).rename('A3_rt_side_left') # left에 따른 평균 시간
    A3_rt_side_right = masked_mean_from_csv_series(df["A3-3"], df["A3-7"], 2).rename('A3_rt_side_right') # right에 따른 평균 시간
    A3_rt_side_diff  = (A3_rt_side_left - A3_rt_side_right).rename('A3_rt_side_diff') # 방향에 따른 평균 시간의 차이
    # ---- 추가 ---- 
    A3_valid_ratio = ((vc + vi) / total).rename('A3_valid_ratio') # 전체 valid
    A3_invalid_ratio = ((ic + ii) / total).rename('A3_invalid_ratio') # 전체 invalid
    A3_correct_ratio = ((vc + ic) / total).rename('A3_correct_ratio') # 전체 correct
    A3_incorrect_ratio = ((vi + ii) / total).rename('A3_incorrect_ratio') # 전체 incorrect
    A3_valid_correct_rate = (vc / (vc + vi + 1e-9)).rename('A3_valid_correct_rate')
    A3_invalid_correct_rate = (ic / (ic + ii + 1e-9)).rename('A3_invalid_correct_rate')
    A3_Q3 = seq_quantile(df['A3-7'], q=0.75).rename('A3_Q3') # 75% 분위수
    A3_Q1 = seq_quantile(df['A3-7'], q=0.25).rename('A3_Q1') # 25% 분위수
    A3_IQR = (A3_Q3 - A3_Q1).rename('A3_IQR') # IQR

    A3_feats = [
        A3_vc_ratio, A3_vi_ratio, A3_ic_ratio, A3_ii_ratio,
        A3_resp2_rate, A3_rt_mean, A3_rt_std, A3_rt_size_small,
        A3_rt_size_big, A3_rt_size_diff, A3_rt_side_left, A3_rt_side_right,
        A3_rt_side_diff,
        A3_valid_ratio, A3_invalid_ratio, A3_correct_ratio, A3_incorrect_ratio,
        A3_valid_correct_rate, A3_invalid_correct_rate, A3_Q3, A3_Q1,
        A3_IQR
    ]

    print(f'A3 feats count: {len(A3_feats)}') # 42

    feats_list.extend(A3_feats)

    # ---- A4 ----
    print('A4 feature 생성')
    A4_acc_rate      = seq_rate(df["A4-3"], "1").rename('A4_acc_rate')
    A4_resp2_rate    = seq_rate(df["A4-4"], "1").rename('A4_resp2_rate')
    A4_rt_mean       = seq_mean(df["A4-5"]).rename('A4_rt_mean')
    A4_rt_std        = seq_std(df["A4-5"]).rename('A4_rt_std')
    A4_stroop_con     = masked_mean_from_csv_series(df["A4-1"], df["A4-5"], 1).rename('A4_stroop_con')
    A4_stroop_incon   = masked_mean_from_csv_series(df["A4-1"], df["A4-5"], 2).rename('A4_stroop_incon')
    A4_stroop_diff    = (A4_stroop_con - A4_stroop_incon).rename('A4_stroop_diff')
    A4_rt_color_red   = masked_mean_from_csv_series(df["A4-2"], df["A4-5"], 1).rename('A4_rt_color_red')
    A4_rt_color_green = masked_mean_from_csv_series(df["A4-2"], df["A4-5"], 2).rename('A4_rt_color_green')
    A4_rt_color_diff  = (A4_rt_color_red - A4_rt_color_green).rename('A4_rt_color_diff')
    # ---- 추가 ----
    A4_correct_rt_mean = masked_mean_from_csv_series(df['A4-3'], df['A4-5'], 1).rename('A4_correct_rt_mean')
    A4_incorrect_rt_mean = masked_mean_from_csv_series(df['A4-3'], df['A4-5'], 2).rename('A4_incorrect_rt_mean')
    A4_resp0_rt_mean = masked_mean_from_csv_series(df['A4-4'], df['A4-5'], 0).rename('A4_resp0_rt_mean')
    A4_resp1_rt_mean = masked_mean_from_csv_series(df['A4-4'], df['A4-5'], 1).rename('A4_resp1_rt_mean')
    A4_correct_rt_std = masked_std_from_csv_series(df['A4-3'], df['A4-5'], 1).rename('A4_correct_rt_std')
    A4_incorrect_rt_std = masked_std_from_csv_series(df['A4-3'], df['A4-5'], 2).rename('A4_incorrect_rt_std')
    A4_resp0_rt_std = masked_std_from_csv_series(df['A4-4'], df['A4-5'], 0).rename('A4_resp0_rt_std')
    A4_resp1_rt_std = masked_std_from_csv_series(df['A4-4'], df['A4-5'], 1).rename('A4_resp1_rt_std')
    A4_rt_median = seq_median(df['A4-5']).rename('A4_rt_median') # 중앙값
    A4_rt_min = seq_min(df['A4-5']).rename('A4_rt_min') # 최솟값
    A4_rt_max = seq_max(df['A4-5']).rename('A4_rt_max') # 최댓값
    A4_rt_skew = seq_skew(df['A4-5']).rename('A4_rt_skew') # 왜도
    A4_rt_kurt = seq_kurt(df['A4-5']).rename('A4_rt_kurt') # 첨도
    A4_diff_mean = seq_diff_mean(df['A4-5']).rename('A4_diff_mean') # 연속된 rt 차이의 평균
    A4_diff_std = seq_diff_std(df['A4-5']).rename('A4_diff_std') # 연속된 rt 차이의 표준편차
    A4_Q3 = seq_quantile(df['A4-5'], q=0.75).rename('A4_Q3') # 75% 분위수
    A4_Q1 = seq_quantile(df['A4-5'], q=0.25).rename('A4_Q1') # 25% 분위수
    A4_IQR = (A4_Q3 - A4_Q1).rename('A4_IQR') # IQR
    A4_rolling_std_mean = seq_rolling_std_mean(df['A4-5']).rename('A4_rolling_std_mean')

    A4_feats = [
        A4_acc_rate, A4_resp2_rate, A4_rt_mean, A4_rt_std,
        A4_stroop_con, A4_stroop_incon, A4_stroop_diff, A4_rt_color_red,
        A4_rt_color_diff, A4_correct_rt_mean, A4_incorrect_rt_mean,
        A4_resp0_rt_mean, A4_resp1_rt_mean, A4_correct_rt_std, 
        A4_incorrect_rt_std, A4_resp0_rt_std, A4_resp1_rt_std,
        A4_rt_median, A4_rt_min, A4_rt_max, A4_rt_skew, 
        A4_rt_kurt, A4_diff_mean, A4_diff_std, A4_Q3, A4_Q1, 
        A4_IQR, A4_rolling_std_mean
    ]

    print(f'A4 feats count: {len(A4_feats)}') # 29

    feats_list.extend(A4_feats)

    # ---- A5 ----
    print('A5 feature 생성')
    A5_acc_rate   = seq_rate(df["A5-2"], "1").rename('A5_acc_rate')
    A5_resp2_rate = seq_rate(df["A5-3"], "1").rename('A5_resp2_rate')
    A5_acc_nonchange = masked_mean_from_csv_series(df["A5-1"], df["A5-2"], 1).rename('A5_acc_nonchange')
    A5_acc_change    = masked_mean_in_set_series(df["A5-1"], df["A5-2"], {2,3,4}).rename('A5_acc_change')

    A5_feats = [
        A5_acc_rate, A5_resp2_rate,
        A5_acc_nonchange, A5_acc_change
    ]

    print(f'A5 feats count: {len(A5_feats)}')

    feats_list.extend(A5_feats)

    # ---- Drop ----
    print('시퀀스 컬럼 drop & concat')
    seq_cols = [
        "A1-1","A1-2","A1-3","A1-4",
        "A2-1","A2-2","A2-3","A2-4",
        "A3-1","A3-2","A3-3","A3-4","A3-5","A3-6","A3-7",
        "A4-1","A4-2","A4-3","A4-4","A4-5",
        "A5-1","A5-2","A5-3"
    ]
    print('A 검사 데이터 전처리 완료')

    feats = pd.concat(feats_list, axis=1)
    return pd.concat([df.drop(columns=seq_cols, errors='ignore'), feats], axis=1)

def preprocess_B(train_B):
    df = train_B.copy()
    original_index = df.index
    # ---- Age, TestDate ----
    print("Step 1: Age, TestDate, PrimaryKey 파생...")
    df["Age_num"] = df["Age"].map(convert_age)
    ym = df["TestDate"].map(split_testdate)
    df["Year"] = [y for y, m in ym]
    df["Month"] = [m for y, m in ym]

    df = df.sort_values(by=['PrimaryKey', 'TestDate']) # Key, Date 순 정렬
    df['past_attempts'] = df.groupby('PrimaryKey').cumcount() # 과거 횟수
    df = df.reindex(original_index)

    feats = pd.DataFrame(index=df.index)

    # ---- B1 ----
    print("Step 2: B1 feature 생성...")
    feats["B1_acc_task1"] = seq_rate(df["B1-1"], "1")
    feats["B1_rt_mean"]   = seq_mean(df["B1-2"])
    feats["B1_rt_std"]    = seq_std(df["B1-2"])
    feats["B1_acc_task2"] = seq_rate(df["B1-3"], "1")

    # ---- B2 ----
    print("Step 3: B2 feature 생성...")
    feats["B2_acc_task1"] = seq_rate(df["B2-1"], "1")
    feats["B2_rt_mean"]   = seq_mean(df["B2-2"])
    feats["B2_rt_std"]    = seq_std(df["B2-2"])
    feats["B2_acc_task2"] = seq_rate(df["B2-3"], "1")

    # ---- B3 ----
    print("Step 4: B3 feature 생성...")
    feats["B3_acc_rate"] = seq_rate(df["B3-1"], "1")
    feats["B3_rt_mean"]  = seq_mean(df["B3-2"])
    feats["B3_rt_std"]   = seq_std(df["B3-2"])

    # ---- B4 ----
    print("Step 5: B4 feature 생성...")
    feats["B4_acc_rate"] = seq_rate(df["B4-1"], "1")
    feats["B4_rt_mean"]  = seq_mean(df["B4-2"])
    feats["B4_rt_std"]   = seq_std(df["B4-2"])

    # ---- B5 ----
    print("Step 6: B5 feature 생성...")
    feats["B5_acc_rate"] = seq_rate(df["B5-1"], "1")
    feats["B5_rt_mean"]  = seq_mean(df["B5-2"])
    feats["B5_rt_std"]   = seq_std(df["B5-2"])

    # ---- B6~B8 ----
    print("Step 7: B6~B8 feature 생성...")
    feats["B6_acc_rate"] = seq_rate(df["B6"], "1")
    feats["B7_acc_rate"] = seq_rate(df["B7"], "1")
    feats["B8_acc_rate"] = seq_rate(df["B8"], "1")

    # ---- Drop ----
    print("Step 8: 시퀀스 컬럼 drop & concat...")
    seq_cols = [
        "B1-1","B1-2","B1-3",
        "B2-1","B2-2","B2-3",
        "B3-1","B3-2",
        "B4-1","B4-2",
        "B5-1","B5-2",
        "B6","B7","B8"
    ]

    print("B 검사 데이터 전처리 완료")
    return pd.concat([df.drop(columns=seq_cols, errors="ignore"), feats], axis=1)
    # return pd.concat([df, feats], axis=1)

def _has(df, cols):  # 필요한 컬럼이 모두 있는지
    return all(c in df.columns for c in cols)

def _safe_div(a, b, eps=1e-6):
    return a / (b + eps)

# -------- A 파생 --------
def add_features_A(df: pd.DataFrame) -> pd.DataFrame:
    feats = df.copy()
    eps = 1e-6

    # 0) Month-Month 단일축
    if _has(feats, ["Year","Month"]):
        feats["YearMonthIndex"] = feats["Year"] * 12 + feats["Month"]

    # 1) 속도-정확도 트레이드오프
    if _has(feats, ["A1_rt_mean","A1_resp_rate"]):
        feats["A1_speed_acc_tradeoff"] = _safe_div(feats["A1_rt_mean"], feats["A1_resp_rate"], eps)
    if _has(feats, ["A2_rt_mean","A2_resp_rate"]):
        feats["A2_speed_acc_tradeoff"] = _safe_div(feats["A2_rt_mean"], feats["A2_resp_rate"], eps)
    if _has(feats, ["A4_rt_mean","A4_acc_rate"]):
        feats["A4_speed_acc_tradeoff"] = _safe_div(feats["A4_rt_mean"], feats["A4_acc_rate"], eps)

    # 2) RT 변동계수(CV)
    for k in ["A1","A2","A3","A4"]:
        m, s = f"{k}_rt_mean", f"{k}_rt_std"
        if _has(feats, [m, s]):
            feats[f"{k}_rt_cv"] = _safe_div(feats[s], feats[m], eps)

    # 3) 조건 차이 절댓값(편향 크기)
    for name, base in [
        ("A1_rt_side_gap_abs",  "A1_rt_side_diff"),
        ("A1_rt_speed_gap_abs", "A1_rt_speed_diff"),
        ("A2_rt_cond1_gap_abs", "A2_rt_cond1_diff"),
        ("A2_rt_cond2_gap_abs", "A2_rt_cond2_diff"),
        ("A4_stroop_gap_abs",   "A4_stroop_diff"),
        ("A4_color_gap_abs",    "A4_rt_color_diff"),
    ]:
        if base in feats.columns:
            feats[name] = feats[base].abs()

    # 4) 정확도 패턴 심화
    if _has(feats, ["A3_valid_ratio","A3_invalid_ratio"]):
        feats["A3_valid_invalid_gap"] = feats["A3_valid_ratio"] - feats["A3_invalid_ratio"]
    if _has(feats, ["A3_correct_ratio","A3_invalid_ratio"]):
        feats["A3_correct_invalid_gap"] = feats["A3_correct_ratio"] - feats["A3_invalid_ratio"]
    if _has(feats, ["A5_acc_change","A5_acc_nonchange"]):
        feats["A5_change_nonchange_gap"] = feats["A5_acc_change"] - feats["A5_acc_nonchange"]

    # 5) 간단 메타 리스크 스코어(휴리스틱)
    parts = []
    if "A4_stroop_gap_abs" in feats: parts.append(0.30 * feats["A4_stroop_gap_abs"].fillna(0))
    if "A4_acc_rate" in feats:       parts.append(0.20 * (1 - feats["A4_acc_rate"].fillna(0)))
    if "A3_valid_invalid_gap" in feats:
        parts.append(0.20 * feats["A3_valid_invalid_gap"].fillna(0).abs())
    if "A1_rt_cv" in feats: parts.append(0.20 * feats["A1_rt_cv"].fillna(0))
    if "A2_rt_cv" in feats: parts.append(0.10 * feats["A2_rt_cv"].fillna(0))
    if parts:
        feats["RiskScore"] = sum(parts)

    # NaN/inf 정리
    feats.replace([np.inf, -np.inf], np.nan, inplace=True)
    return feats

# -------- B 파생 --------
def add_features_B(df: pd.DataFrame) -> pd.DataFrame:
    feats = df.copy()
    eps = 1e-6

    # 0) Year-Month 단일축
    if _has(feats, ["Year","Month"]):
        feats["YearMonthIndex"] = feats["Year"] * 12 + feats["Month"]

    # 1) 속도-정확도 트레이드오프 (B1~B5)
    for k, acc_col, rt_col in [
        ("B1", "B1_acc_task1", "B1_rt_mean"),
        ("B2", "B2_acc_task1", "B2_rt_mean"),
        ("B3", "B3_acc_rate",  "B3_rt_mean"),
        ("B4", "B4_acc_rate",  "B4_rt_mean"),
        ("B5", "B5_acc_rate",  "B5_rt_mean"),
    ]:
        if _has(feats, [rt_col, acc_col]):
            feats[f"{k}_speed_acc_tradeoff"] = _safe_div(feats[rt_col], feats[acc_col], eps)

    # 2) RT 변동계수(CV)
    for k in ["B1","B2","B3","B4","B5"]:
        m, s = f"{k}_rt_mean", f"{k}_rt_std"
        if _has(feats, [m, s]):
            feats[f"{k}_rt_cv"] = _safe_div(feats[s], feats[m], eps)

    # 3) 간단 메타 리스크 스코어(휴리스틱)
    parts = []
    for k in ["B4","B5"]:  # 주의집중/스트룹 유사 과제 가중
        if _has(feats, [f"{k}_rt_cv"]):
            parts.append(0.25 * feats[f"{k}_rt_cv"].fillna(0))
    for k in ["B3","B4","B5"]:
        acc = f"{k}_acc_rate" if k != "B1" and k != "B2" else None
        if k in ["B1","B2"]:
            acc = f"{k}_acc_task1"
        if acc in feats:
            parts.append(0.25 * (1 - feats[acc].fillna(0)))
    for k in ["B1","B2"]:
        tcol = f"{k}_speed_acc_tradeoff"
        if tcol in feats:
            parts.append(0.25 * feats[tcol].fillna(0))
    if parts:
        feats["RiskScore_B"] = sum(parts)

    feats.replace([np.inf, -np.inf], np.nan, inplace=True)
    return feats

# =======================
# 정렬/보정 (모델이 학습 때 본 피처 순서로)
# =======================
DROP_COLS = ["Test_id","Test","PrimaryKey","Age","TestDate", "Test_x", 'Test_y', 'Label']

def align_to_model(X_df, model_list):
    # 앙상블의 첫 번째 모델을 기준으로 피처 목록을 가져옵니다.
    base_hist_model = model_list[0].calibrated_classifiers_[0].estimator
    # .feature_name_ -> .feature_names_in_ 으로 수정
    feat_names = base_hist_model.feature_names_in_

    X = X_df.drop(columns=[c for c in DROP_COLS if c in X_df.columns], errors="ignore").copy()
    
    # 학습 때 사용되었으나 현재 데이터에 없는 피처는 0으로 채웁니다.
    for c in feat_names:
        if c not in X.columns:
            X[c] = 0
            
    # 학습 때 사용된 피처 순서대로 컬럼을 정렬하고, 없는 컬럼은 버립니다.
    return X[feat_names]

# 1.1. FINAL 모델(1단계) 앙상블 함수
def predict_final_ensemble(model_list, X_data):
    if not model_list:
        return np.array([])
    all_predictions = []
    for model in model_list:
        all_predictions.append(np.clip(model.predict_proba(X_data)[:, 1], 1e-7, 1-1e-7))
    # 3개 Seed 모델의 예측값을 평균
    return np.mean(all_predictions, axis=0)

# 1.2. META 모델(2단계) 예측 함수
def predict_meta(meta_model, X_meta_data):
    if not len(X_meta_data):
        return np.array([])
    # 1단계 예측 결과(1D 배열)를 2D DataFrame으로 변환
    X_meta_df = pd.DataFrame({'OOF_Pred': X_meta_data})
    # 2단계 모델로 최종 확률 예측
    return meta_model.predict_proba(X_meta_df)[:, 1]

import glob

def main():
    # ---- 경로 변수 (필요에 따라 수정) ----
    TEST_DIR  = "./data"              # test.csv, A.csv, B.csv, sample_submission.csv 위치
    MODEL_DIR = "./model"             # lgbm_A.pkl, lgbm_B.pkl 위치
    OUT_DIR   = "./output"
    SAMPLE_SUB_PATH = os.path.join(TEST_DIR, "sample_submission.csv")
    OUT_PATH  = os.path.join(OUT_DIR, "submission.csv")

    # ---- 모델 로드 ----
    print("Load 1-Level (FINAL) and 2-Level (META) models...")
    MODEL_PATH_A_FINAL = os.path.join(MODEL_DIR, "model_A_FINAL.pkl")
    MODEL_PATH_B_FINAL = os.path.join(MODEL_DIR, "model_B_FINAL.pkl")
    MODEL_PATH_A_META = os.path.join(MODEL_DIR, "model_A_META.pkl")
    MODEL_PATH_B_META = os.path.join(MODEL_DIR, "model_B_META.pkl")

    model_A_final = joblib.load(MODEL_PATH_A_FINAL) # 1단계 A (리스트)
    model_B_final = joblib.load(MODEL_PATH_B_FINAL) # 1단계 B (리스트)
    model_A_meta = joblib.load(MODEL_PATH_A_META)   # 2단계 A (단일 모델)
    model_B_meta = joblib.load(MODEL_PATH_B_META)   # 2단계 B (단일 모델)
    
    print(" OK.")

    # ---- 테스트 데이터 로드 ----
    print("Load test data...")
    meta = pd.read_csv(os.path.join(TEST_DIR, "test.csv"))
    Araw = pd.read_csv(os.path.join(TEST_DIR, "./test/A.csv"))
    Braw = pd.read_csv(os.path.join(TEST_DIR, "./test/B.csv"))
    print(f" meta={len(meta)}, Araw={len(Araw)}, Braw={len(Braw)}")

    # ---- 매핑 ----
    A_df = meta.loc[meta["Test"] == "A", ["Test_id", "Test"]].merge(Araw, on="Test_id", how="left")
    B_df = meta.loc[meta["Test"] == "B", ["Test_id", "Test"]].merge(Braw, on="Test_id", how="left")
    print(f" mapped: A={len(A_df)}, B={len(B_df)}")

    # ---- 전처리 → 파생 (학습과 동일) ----

    A_feat = preprocess_A(A_df)
    B_feat = preprocess_B(B_df)
    A_feat = add_features_A(A_feat)
    B_feat = add_features_B(B_feat)

    # ---- 피처 정렬/보정 ----
    XA = align_to_model(A_feat, model_A_final) if len(A_feat) else pd.DataFrame(columns=getattr(model_A_final,"feature_name_",[]))
    XB = align_to_model(B_feat, model_B_final) if len(B_feat) else pd.DataFrame(columns=getattr(model_B_final,"feature_name_",[]))
    print(f" aligned: XA={XA.shape}, XB={XB.shape}")

    # ---- 예측 ----
    print("Inference Model...")
    predA_M1 = predict_final_ensemble(model_A_final, XA) # 1단계 예측
    predB_M1 = predict_final_ensemble(model_B_final, XB) # 1단계 예측

    print("Inference 2-Level (META) Model...")
    predA_M2 = predict_meta(model_A_meta, predA_M1) # 2단계 (최종) 예측
    predB_M2 = predict_meta(model_B_meta, predB_M1) # 2단계 (최종) 예측

    # ---- Test_id와 합치기 ----
    subA = pd.DataFrame({"Test_id": A_df["Test_id"].values, "prob": predA_M2})
    subB = pd.DataFrame({"Test_id": B_df["Test_id"].values, "prob": predB_M2})
    probs = pd.concat([subA, subB], axis=0, ignore_index=True)
    # final_submission_df = probs.rename(columns={'prob': 'Label'})
    # answer_df_A = A_df[['Test_id', 'Label']]
    # answer_df_B = B_df[['Test_id', 'Label']]
    # final_answer_df = pd.concat([answer_df_A, answer_df_B], ignore_index=True)
    # final_score = auc_brier_ece(final_answer_df, final_submission_df)
    # print("\n" + "="*50)
    # print(f"✅ TOTAL COMBINED SCORE (on full train data): {final_score:.6f}")
    # print("="*50)

    # ---- sample_submission 기반 결과 생성 (Label 컬럼에 0~1 확률 채움) ----
    os.makedirs(OUT_DIR, exist_ok=True)
    sample = pd.read_csv(SAMPLE_SUB_PATH)
    # sample의 Test_id 순서에 맞추어 prob 병합
    out = sample.merge(probs, on="Test_id", how="left")
    out["Label"] = out["prob"].astype(float).fillna(0.0)
    out = out.drop(columns=["prob"])

    out.to_csv(OUT_PATH, index=False) # [수정] 주석 해제
    print(f"✅ Saved: {OUT_PATH} (rows={len(out)})")

if __name__ == "__main__":
    main()
