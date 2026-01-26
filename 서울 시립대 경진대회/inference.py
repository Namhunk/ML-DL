import yaml
import joblib
import optuna
import numpy as np
import pandas as pd
from tqdm import tqdm
from autogluon.tabular import TabularPredictor

# Data Load 함수 file_path: 파일 이름, folder_path: 데이터가 저장된 folder 위치
def load_data(file_path, folder_path = './open_track1/'): return pd.read_csv(folder_path+file_path)

# feature 생성 함수 
def create_feature(raw_df, player_map, team_map, grid_map, global_map, point, type_med, window=3, is_train=True):
    df = raw_df.copy() # 원본 데이터 복사

    df['result_name'] = df['result_name'].fillna('Unknown') # result_name의 nan -> "Unknown"

    # 각 x, y 크기 정규화(0 <= x,y <= 1)
    df['start_x'] /= 105.0; df['end_x'] /= 105.0
    df['start_y'] /= 68.0;  df['end_y'] /= 68.0

    # game_episode, time_seconds 순서로 정렬
    df = df.sort_values(by=['game_episode', 'time_seconds']).reset_index(drop=True)

    feats = [] # 각 game_episode 별로 생성한 feature들의 묶음을 저장할 배열

    # df를 game_episode로 그룹화
    grouped = df.groupby('game_episode')

    for game_episode, group in tqdm(grouped, desc='Feature Extraction'):
        feature_dict = {} # 각 game_episode 마다 생성한 feature를 저장할 딕셔너리
        group = group.reset_index(drop=True)

        feature_dict['game_episode'] = game_episode # 현재 game_episode

        # 가장 마지막 이벤트에 대한 데이터
        last_row = group.iloc[-1] # 마지막 행을 추출

        feature_dict['last_start_x']      = last_row['start_x']      # x좌표
        feature_dict['last_start_y']      = last_row['start_y']      # y좌표
        feature_dict['last_player_id']    = last_row['player_id']    # 선수 ID
        feature_dict['last_team_id']      = last_row['team_id']      # 팀 ID
        feature_dict['last_type_name']    = last_row['type_name']    # 행동 타입
        feature_dict['last_result_name']  = last_row['result_name']  # 성공 여부
        feature_dict['last_time_seconds'] = last_row['time_seconds'] # 시간
        feature_dict['last_is_home']      = last_row['is_home']      # 홈 팀 여부

        # 이전 이벤트에 대한 데이터
        past_group = group.iloc[:-1].copy() # 마지막 이벤트를 제외한 나머지 그룹

        if len(past_group) > 0: # 시퀀스의 길이가 1보다 큰 경우만
            prev_row = past_group.iloc[-1] # 마지막 시점 이전의 데이터

            # 마지막 시작 위치와 이전 종료 위치의 차이(x, y)
            feature_dict['prev_last_dx']   = last_row['start_x'] - prev_row['end_x'] # x 차이
            feature_dict['prev_last_dy']   = last_row['start_y'] - prev_row['end_y'] # y 차이
            feature_dict['prev_last_dist'] = np.sqrt(feature_dict['prev_last_dx']**2 + feature_dict['prev_last_dy']**2) # 거리 계산

            # 이전 이벤트에 대한 데이터
            feature_dict['prev_start_x']      = prev_row['start_x']      # start x좌표
            feature_dict['prev_start_y']      = prev_row['start_y']      # start y좌표
            feature_dict['prev_end_x']        = prev_row['end_x']        # end x좌표
            feature_dict['prev_end_y']        = prev_row['end_y']        # end y좌표
            feature_dict['prev_player_id']    = prev_row['player_id']    # 선수 ID
            feature_dict['prev_team_id']      = prev_row['team_id']      # 팀 ID
            feature_dict['prev_type_name']    = prev_row['type_name']    # 행동 타입
            feature_dict['prev_result_name']  = prev_row['result_name']  # 성공여부
            feature_dict['prev_time_seconds'] = prev_row['time_seconds'] # 시간
            feature_dict['prev_is_home']      = prev_row['is_home']      # 홈 팀 여부

            # feature 조합 및 연산을 통한 데이터 생성
            feature_dict['prev_dx']    = prev_row['end_x'] - prev_row['start_x'] # 이전 이벤트의 dx
            feature_dict['prev_dy']    = prev_row['end_y'] - prev_row['start_y'] # 이전 이벤트의 dy
            feature_dict['last_dt']    = last_row['time_seconds'] - prev_row['time_seconds'] # 마지막/이전 이벤트의 시간 차이
            feature_dict['prev_dist']  = np.sqrt(feature_dict['prev_dx']**2 + feature_dict['prev_dy']**2) # 이전 이벤트의 거리
            feature_dict['prev_speed'] = feature_dict['prev_dist'] / (feature_dict['last_dt'] + 1e-8) # 이전 이벤트의 속도
            feature_dict['prev_sin']   = feature_dict['prev_dy'] / (feature_dict['prev_dist'] + 1e-8) # 이전 이벤트의 sin
            feature_dict['prev_cos']   = feature_dict['prev_dx'] / (feature_dict['prev_dist'] + 1e-8) # 이전 이벤트의 cos

            # prev의 dx, dy에 시간을 나눔
            feature_dict['prev_dx_dt_div'] = feature_dict['prev_dx'] / (feature_dict['last_dt'] + 1e-8)
            feature_dict['prev_dy_dt_div'] = feature_dict['prev_dy'] / (feature_dict['last_dt'] + 1e-8)

            # prev의 dx, dy에 속도를 곱함
            feature_dict['prev_dx_speed'] = feature_dict['prev_dx'] * feature_dict['prev_speed']
            feature_dict['prev_dy_speed'] = feature_dict['prev_dy'] * feature_dict['prev_speed']

            # prev의 dx, dy abs값에 속도를 곱함
            feature_dict['prev_dx_speed_abs'] = abs(feature_dict['prev_dx']) * feature_dict['prev_speed']
            feature_dict['prev_dy_speed_abs'] = abs(feature_dict['prev_dy']) * feature_dict['prev_speed']

            # prev_last의 dx, dy에 시간을 나눔
            feature_dict['prev_last_x_vel'] = feature_dict['prev_last_dx'] / (feature_dict['last_dt'] + 1e-8)
            feature_dict['prev_last_y_vel'] = feature_dict['prev_last_dy'] / (feature_dict['last_dt'] + 1e-8)
            
            # prev_last의 dx, dy에 시간을 곱함
            feature_dict['prev_last_x_dt'] = feature_dict['prev_last_dx'] * feature_dict['last_dt']
            feature_dict['prev_last_y_dt'] = feature_dict['prev_last_dy'] * feature_dict['last_dt']

            # player_map을 이용한 feature 생성
            last_player_map = player_map.get(last_row['player_id'], player_map['global'])
            prev_player_map = player_map.get(prev_row['player_id'], player_map['global'])

            feature_dict['last_player_dy'] = last_player_map['player_mean_dy'] # 마지막 선수의 dy 평균
            feature_dict['prev_player_dy'] = prev_player_map['player_mean_dy'] # 이전 선수의 dy 평균

            feature_dict['ll_player_dy']         = last_player_map['player_start_y'] - feature_dict['last_start_y'] # 
            feature_dict['prev_player_dx_diff']  = feature_dict['prev_last_dx'] - prev_player_map['player_mean_dx'] # (last_start_x - prev_end_x) - (player_mean_start_y)
            feature_dict['plsx_pey_diff']        = last_player_map['player_start_x'] - feature_dict['prev_end_y']   # 마지막 선수의 평균 start_x - 이전 선수의 end_y

            feature_dict['last_player_dy_range'] = last_player_map['dy_range'] # 마지막 선수의 dy 범위
            
            # 마지막 선수의 dy/dx * 이전 선수의 dy/dx
            feature_dict['player_dy_dx_div']     = (last_player_map['player_mean_dy'] * prev_player_map['player_mean_dy']) / (last_player_map['player_mean_dx'] * prev_player_map['player_mean_dx'])
            
            # grid 데이터
            last_x_idx = min(int(feature_dict['last_start_x']*5), 4) # x 위치
            last_y_idx = min(int(feature_dict['last_start_y']*5), 4) # y 위치
            last_p     = feature_dict['last_player_id']              # 선수 번호

            key = (last_p, last_x_idx, last_y_idx) # key

            if grid_map and key in grid_map: # key의 정보가 있는 경우
                stats = grid_map[key]
                feature_dict['grid_dx'] = stats['dx']
                feature_dict['grid_dy'] = stats['dy']
            elif global_map and last_p in global_map: # 선수 정보만 있는 경우
                stats = global_map[last_p]
                feature_dict['grid_dx'] = stats['dx']
                feature_dict['grid_dy'] = stats['dy']
            else: # 둘다 없는 경우
                feature_dict['grid_dx'] = 0.0
                feature_dict['grid_dy'] = 0.0
            
            feature_dict['last_result_grid_dx'] = f"{feature_dict['last_result_name']}_{feature_dict['grid_dx']}" # 마지막 결과와 grid_dx
            feature_dict['prev_type_grid_dy']   = f"{feature_dict['prev_type_name']}_{feature_dict['grid_dy']}"   # 이전 행동과 grid_dy  
            feature_dict['prev_type_grid_dx']   = f"{feature_dict['prev_type_name']}_{feature_dict['grid_dx']}"   # 이전 행동과 grid_dx

            # team에 대한 데이터
            last_team_map = team_map.get(last_row['team_id'], team_map['global'])
            prev_team_map = team_map.get(prev_row['team_id'], team_map['global'])

            feature_dict['last_team_start_x'] = last_team_map['team_start_x']
            feature_dict['prev_team_start_x'] = prev_team_map['team_start_x']

            # quantile 정보
            feature_dict['g_start_x_75'] = group['start_x'].quantile(0.75)
            feature_dict['g_start_y_25'] = group['start_y'].quantile(0.25)

            # 이동 평균
            if len(past_group) >= window: # window 크기보다 큰 경우만
                past_group['step_dx']    = past_group['end_x'] - past_group['start_x']
                past_group['step_dy']    = past_group['end_y'] - past_group['start_y']
                past_group['step_dist']  = np.sqrt(past_group['step_dx']**2 + past_group['step_dy']**2)
                past_group['step_sin']   = past_group['step_dy'] / (past_group['step_dist'] + 1e-8)
                past_group['step_cos']   = past_group['step_dx'] / (past_group['step_dist'] + 1e-8)
                past_group['time_diff']  = past_group['time_seconds'].diff().fillna(0)
                past_group['step_speed'] = np.where(past_group['time_diff'] > 0, past_group['step_dist'] / past_group['time_diff'], 0)

                # 과거의 마지막 window 만큼의 평균
                feature_dict['roll_start_x_mean'] = past_group['start_x'].tail(window).mean()
                feature_dict['roll_start_y_mean'] = past_group['start_y'].tail(window).mean()
                feature_dict['roll_end_x_mean']   = past_group['end_x'].tail(window).mean()
                feature_dict['roll_end_y_mean']   = past_group['end_y'].tail(window).mean()
                feature_dict['roll_dx_mean']      = past_group['step_dx'].tail(window).mean()
                feature_dict['roll_dy_mean']      = past_group['step_dy'].tail(window).mean()
                feature_dict['roll_dist_mean']    = past_group['step_dist'].tail(window).mean()
                feature_dict['roll_sin_mean']     = past_group['step_sin'].tail(window).mean()
                feature_dict['roll_cos_mean']     = past_group['step_cos'].tail(window).mean()
                feature_dict['roll_speed_med']    = past_group['step_speed'].tail(window).median()

                # 과거의 마지막 window 만큼의 변동계수
                feature_dict['roll_dx_cv']       = past_group['step_dx'].tail(window).std() / (feature_dict['roll_dx_mean'] + 1e-6)
                feature_dict['roll_dy_cv']       = past_group['step_dy'].tail(window).std() / (feature_dict['roll_dy_mean'] + 1e-6)
                feature_dict['roll_dist_cv']     = past_group['step_dist'].tail(window).std() / (feature_dict['roll_dist_mean'] + 1e-8)
                feature_dict['roll_speed_cv']    = past_group['step_speed'].tail(window).std() / (feature_dict['roll_speed_med'] + 1e-8)
                feature_dict['roll_sin_cv']      = past_group['step_sin'].tail(window).std() / (feature_dict['roll_sin_mean'] + 1e-8)
                feature_dict['roll_cos_cv']      = past_group['step_cos'].tail(window).std() / (feature_dict['roll_cos_mean'] + 1e-8)

                # window dy 평균 - 이전 dy
                feature_dict['roll_prev_dy']     = feature_dict['roll_dy_mean'] - feature_dict['prev_dy']

                # 벡터 내적
                vec_x_1 = feature_dict['prev_last_dx']
                vec_x_2 = feature_dict['prev_dx']
                vec_y_1 = feature_dict['prev_last_dy']
                vec_y_2 = feature_dict['prev_dy']
                feature_dict['prev_last_dot_prod'] = (vec_x_1 * vec_x_2) + (vec_y_1 * vec_y_2)

                # 이전 그룹의 start_x, start_y의 상관계수
                feature_dict['start_xy_corr'] = past_group['start_x'].corr(past_group['start_y'])

                # quantile 데이터
                feature_dict['pg_dx_25']     = past_group['step_dx'].quantile(0.25)
                feature_dict['pg_dy_25']     = past_group['step_dy'].quantile(0.25)
                feature_dict['pg_dy_75']     = past_group['step_dy'].quantile(0.75)
                feature_dict['window_dx_25'] = past_group['step_dx'].tail(window).quantile(0.25)

            else: # 과거 데이터 부족 시 직전 값
                feature_dict['roll_start_x_mean'] = feature_dict['prev_start_x']
                feature_dict['roll_start_y_mean'] = feature_dict['prev_start_y']
                feature_dict['roll_end_x_mean']   = feature_dict['prev_end_x']
                feature_dict['roll_end_y_mean']   = feature_dict['prev_end_y']
                feature_dict['roll_dx_mean']      = feature_dict['prev_dx']
                feature_dict['roll_dy_mean']      = feature_dict['prev_dy']
                feature_dict['roll_dist_mean']    = feature_dict['prev_dist']
                feature_dict['roll_speed_med']    = feature_dict['prev_speed']
                feature_dict['roll_sin_mean']     = feature_dict['prev_sin']
                feature_dict['roll_cos_mean']     = feature_dict['prev_cos']

                feature_dict['roll_dx_cv']       = 0.0
                feature_dict['roll_dy_cv']       = 0.0
                feature_dict['roll_dist_cv']     = 0.0
                feature_dict['roll_speed_cv']    = 0.0
                feature_dict['roll_sin_cv']      = 0.0
                feature_dict['roll_cos_cv']      = 0.0
                feature_dict['roll_prev_dy']     = np.nan

                feature_dict['prev_last_dot_prod'] = np.nan
                feature_dict['start_xy_corr']      = 0

                feature_dict['pg_dx_25']           = np.nan
                feature_dict['pg_dy_25']           = np.nan
                feature_dict['pg_dy_75']           = np.nan
                feature_dict['window_dx_25']       = np.nan

        else: # 과거 기록 없음
            feature_dict['prev_last_dx']     = 0.0
            feature_dict['prev_last_dy']     = 0.0
            feature_dict['prev_last_dist']   = 0.0

            feature_dict['prev_start_x']      = 0.0
            feature_dict['prev_start_y']      = 0.0
            feature_dict['prev_end_x']        = 0.0
            feature_dict['prev_end_y']        = 0.0
            feature_dict['prev_player_id']    = np.nan
            feature_dict['prev_team_id']      = np.nan
            feature_dict['prev_type_name']    = np.nan
            feature_dict['prev_result_name']  = np.nan
            feature_dict['prev_time_seconds'] = np.nan
            feature_dict['prev_is_home']      = np.nan

            feature_dict['prev_dx']    = 0.0
            feature_dict['prev_dy']    = 0.0
            feature_dict['last_dt']    = 0.0
            feature_dict['prev_dist']  = 0.0
            feature_dict['prev_speed'] = 0.0
            feature_dict['prev_sin']   = 0.0
            feature_dict['prev_cos']   = 0.0

            feature_dict['prev_dx_dt_div'] = 0.0 
            feature_dict['prev_dy_dt_div'] = 0.0

            feature_dict['prev_dx_speed']  = 0.0
            feature_dict['prev_dy_speed']  = 0.0

            feature_dict['prev_dx_speed_abs'] = 0.0 
            feature_dict['prev_dy_speed_abs'] = 0.0

            feature_dict['prev_last_x_vel']   = 0.0
            feature_dict['prev_last_y_vel']   = 0.0

            feature_dict['prev_last_x_dt']    = 0.0
            feature_dict['prev_last_y_dt']    = 0.0

            feature_dict['last_player_dy']       = 0.0
            feature_dict['prev_player_dy']       = 0.0
            feature_dict['ll_player_dy']         = 0.0
            feature_dict['prev_player_dx_diff']  = 0.0 
            feature_dict['plsx_pey_diff']        = 0.0 
            feature_dict['last_player_dy_range'] = 0.0
            feature_dict['player_dy_dx_div']     = 0.0

            feature_dict['grid_dx'] = 0.0
            feature_dict['grid_dy'] = 0.0

            feature_dict['last_result_grid_dx'] = np.nan
            feature_dict['prev_type_grid_dy']   = np.nan
            feature_dict['prev_type_grid_dx']   = np.nan

            feature_dict['last_team_start_x'] = 0.0
            feature_dict['prev_team_start_x'] = 0.0

            # quantile 정보
            feature_dict['g_start_x_75'] = 0.0
            feature_dict['g_start_y_25'] = 0.0

            feature_dict['roll_start_x_mean'] = feature_dict['prev_start_x']
            feature_dict['roll_start_y_mean'] = feature_dict['prev_start_y']
            feature_dict['roll_end_x_mean']   = feature_dict['prev_end_x']
            feature_dict['roll_end_y_mean']   = feature_dict['prev_end_y']
            feature_dict['roll_dx_mean']      = feature_dict['prev_dx']
            feature_dict['roll_dy_mean']      = feature_dict['prev_dy']
            feature_dict['roll_dist_mean']    = feature_dict['prev_dist']
            feature_dict['roll_speed_med']    = feature_dict['prev_speed']
            feature_dict['roll_sin_mean']     = feature_dict['prev_sin']
            feature_dict['roll_cos_mean']     = feature_dict['prev_cos']
        
            feature_dict['roll_dx_cv']       = 0.0
            feature_dict['roll_dy_cv']       = 0.0
            feature_dict['roll_dist_cv']     = 0.0
            feature_dict['roll_speed_cv']    = 0.0
            feature_dict['roll_sin_cv']      = 0.0
            feature_dict['roll_cos_cv']      = 0.0
            feature_dict['roll_prev_dy']     = np.nan
        
            feature_dict['prev_last_dot_prod'] = np.nan
            feature_dict['start_xy_corr']      = 0
        
            feature_dict['pg_dx_25']           = 0.0
            feature_dict['pg_dy_25']           = 0.0
            feature_dict['pg_dy_75']           = 0.0
            feature_dict['window_dx_25']       = 0.0

        # 필드에서 dx, dy와 상관계수가 높은 포인트 4
        for i, (x, y) in enumerate(point):
            x /= 105.0
            y /= 68.0

            dx = feature_dict['last_start_x'] - x
            dy = feature_dict['last_start_y'] - y

            feature_dict[f'field_dx_{i}'] = dx
            feature_dict[f'field_dy_{i}'] = dy

            prev_dx = feature_dict['prev_start_x'] - x
            prev_dy = feature_dict['prev_start_y'] - y

            feature_dict[f'field_dist_{i}'] = np.sqrt(dx**2 + dy**2) # 거리
            feature_dict[f'prev_field_dist_{i}'] = np.sqrt(prev_dx**2 + prev_dy**2) # 이전 위치와의 거리
            feature_dict[f'last_prev_field_dist_diff_{i}'] = feature_dict[f'field_dist_{i}'] - feature_dict[f'prev_field_dist_{i}']

            if i == 1:
                feature_dict[f'field_dx_dt_{i}'] = dx * feature_dict['last_dt']
                feature_dict[f'field_dy_dt_{i}'] = dy * feature_dict['last_dt']
            
            if i < 3:
                feature_dict[f'field_speed_{i}'] = feature_dict[f'field_dist_{i}'] / (feature_dict['last_dt'] + 1e-8)
            
            if i > 0:
                feature_dict[f'prev_field_speed_{i}'] = feature_dict[f'prev_field_dist_{i}'] / (feature_dict['last_dt'] + 1e-8)
            
            if 0 < i < 3:
                feature_dict[f'field_speed_diff_{i}'] = feature_dict[f'field_speed_{i}'] - feature_dict[f'prev_field_speed_{i}']

            if i == 3:
                feature_dict[f'prev_field_dx_speed_{i}'] = prev_dx * feature_dict['prev_speed']
            
        # 벡터 내적
        for i, (x, y) in enumerate(point):
            vec_v_x = feature_dict['prev_dx']
            vec_v_y = feature_dict['prev_dy']

            vec_t_x = x - feature_dict['last_start_x']
            vec_t_y = y - feature_dict['last_start_y']
            
            t_norm = np.sqrt(vec_t_x**2 + vec_t_y**2) + 1e-8
            unit_t_x = vec_t_x / t_norm
            unit_t_y = vec_t_y / t_norm

            feature_dict[f'dot_prod_{i}'] = (vec_v_x * unit_t_x) + (vec_v_y * unit_t_y)

        # 외적 (0 번 좌표에 대해서만)
        vec_v_x = feature_dict['prev_dx']
        vec_v_y = feature_dict['prev_dy']

        vec_t_x = 50 - feature_dict['last_start_x']
        vec_t_y = 68 - feature_dict['last_start_y']

        v_norm  = np.sqrt(vec_v_x**2 + vec_v_y**2) + 1e-8
        t_norm = np.sqrt(vec_t_x**2 + vec_t_y**2) + 1e-8
        unit_t_x = vec_t_x / t_norm
        unit_t_y = vec_t_y / t_norm
        feature_dict['cross_prod_0'] = (vec_v_x * unit_t_y) - (vec_v_y * unit_t_x)
        feature_dict['cos_theta_0'] = feature_dict['dot_prod_0'] / v_norm

        # 중앙관련 
        feature_dict['center_dx'] = (0.5 - feature_dict['last_start_x'])
        feature_dict['center_dy'] = (0.5 - feature_dict['last_start_y'])
        feature_dict['center_dy_abs'] = abs(feature_dict['center_dy'])
        feature_dict['center_dx_abs'] = abs(feature_dict['center_dx'])

        feature_dict['center_dx_dt'] = feature_dict['center_dx'] * feature_dict['last_dt']
        feature_dict['center_dy_dt'] = feature_dict['center_dy'] * feature_dict['last_dt']

        feature_dict['dist_to_center'] = np.sqrt(feature_dict['center_dx']**2 + feature_dict['center_dy']**2) # 중앙까지의 거리

        feature_dict['dist_to_center_speed'] = feature_dict['dist_to_center']  / (feature_dict['prev_speed'] + 1e-8)

        PITCH_LEN_X, PITCH_WID_Y = 105.0, 68.0

        # 규격 정의
        norm_pen_depth = 16.5 / PITCH_LEN_X      # 패널티 박스 깊이
        norm_pen_width_half = (40.32 / 2) / PITCH_WID_Y # 패널티 박스 폭 절반

        pen_box_x_start = 1.0 - norm_pen_depth
        pen_box_y_min = 0.5 - norm_pen_width_half
        pen_box_y_max = 0.5 + norm_pen_width_half

        # 진입 여부 판별
        in_pen_box = 1 if (feature_dict['last_start_x'] >= pen_box_x_start) and (pen_box_y_min <= feature_dict['last_start_y'] <= pen_box_y_max) else 0

        if in_pen_box:
            feature_dict['dist_to_pen_edge'] = 0.0
        else:
            dx_pen = max(0, pen_box_x_start - feature_dict['last_start_x'])
            if feature_dict['last_start_y'] < pen_box_y_min:
                dy_pen = pen_box_y_min - feature_dict['last_start_y']
            elif feature_dict['last_start_y'] > pen_box_y_max:
                dy_pen = feature_dict['last_start_y'] - pen_box_y_max
            else:
                dy_pen = 0
            feature_dict['dist_to_pen_edge'] = np.sqrt(dx_pen**2 + dy_pen**2)
        
        goal_x, goal_y = 1.0, 0.5
        vec_goal_x = goal_x - feature_dict['last_start_x']
        vec_goal_y = goal_y - feature_dict['last_start_y']

        feature_dict['dist_to_goal_center'] = np.sqrt(vec_goal_x**2 + vec_goal_y**2)
        feature_dict['angle_to_goal'] = np.arctan2(vec_goal_y, vec_goal_x)
        post1_y = 0.5 - (3.66 / 68.0)
        post2_y = 0.5 + (3.66 / 68.0)

        vec_p1_x, vec_p1_y = 1.0 - feature_dict['last_start_x'], post1_y - feature_dict['last_start_y']
        vec_p2_x, vec_p2_y = 1.0 - feature_dict['last_start_x'], post2_y - feature_dict['last_start_y']
        angle_p1 = np.arctan2(vec_p1_y, vec_p1_x)
        angle_p2 = np.arctan2(vec_p2_y, vec_p2_x)
        feature_dict['visible_goal_angle'] = abs(angle_p1 - angle_p2)

        # meta 정보
        feature_dict['home_team']  = last_row['home_team_id']
        feature_dict['away_team']  = last_row['away_team_id']
        feature_dict['home_score'] = last_row['home_score']
        feature_dict['away_score'] = last_row['away_score']
        feature_dict['prev_type_result'] = f'{prev_row['type_name']}_{prev_row['result_name']}'

        # nan값이 아닌 경우만
        if pd.notna(feature_dict.get('prev_type_name')) and pd.notna(feature_dict.get('prev_dist')):
            name = feature_dict['prev_type_name']
            prev_dist = feature_dict['prev_dist']
            typical_dist = type_med.get(name, 0)
            feature_dict['type_med_diff'] = prev_dist - typical_dist
        else:
            feature_dict['type_med_diff'] = 0  # 또는 np.nan
    
        if is_train:
            feature_dict['target_dx']  = last_row['end_x'] - last_row['start_x'] # 마지막 x위치 변화
            feature_dict['target_dy']  = last_row['end_y'] - last_row['start_y'] # 마지막 y위치 변화
            feature_dict['true_end_x'] = last_row['end_x'] # 실제 x위치
            feature_dict['true_end_y'] = last_row['end_y'] # 실제 y위치

        feats.append(feature_dict)
    
    return pd.DataFrame(feats)

# dx, dy에 따라 drop할 feature 선택
def drop_columns(target):
    if target == 'dx':
        drop_cols = ['game_episode']
        drop_cols += ['last_player_dy', 'plsx_pey_diff', 'grid_dy', 'visible_goal_angle', 'angle_to_goal', 'type_med_diff']
        drop_cols += ['start_xy_corr', 'g_start_x_75', 'g_start_y_25', 'pg_dx_25', 'window_dx_25','pg_dy_25']
        drop_cols += ['field_dy_dt_1', 'prev_field_speed_1', 'field_speed_0', 'field_speed_1', 'field_speed_2', 'prev_field_speed_3']
        drop_cols += ['prev_last_x_dt', 'field_speed_diff_1', 'field_speed_diff_2', 'prev_field_dx_speed_3']
        drop_cols += ['prev_type_grid_dy']
    else:
        drop_cols = ['game_episode']
        drop_cols += ['player_dy_dx_div','last_team_start_x', 'prev_team_start_x','pg_dy_75', 'cross_prod_0', 'cos_theta_0']
        drop_cols += ['prev_last_x_vel', 'prev_last_y_vel', 'center_dy_abs', 'center_dx_abs', 'prev_type_result', 'prev_dx_dt_div', 'prev_dy_dt_div']
        drop_cols += ['prev_dx_speed', 'prev_dy_speed', 'prev_dx_speed_abs', 'prev_dy_speed_abs', 'field_dx_dt_1']
        drop_cols += ['prev_field_speed_2', 'prev_last_y_dt', 'center_dx_dt', 'center_dy_dt', 'dist_to_center_speed']
        drop_cols += ['grid_dx', 'last_result_grid_dx']
    
    return drop_cols

# pred 값이 lower보다 작으면 0, upper보다 크면 max_weight, 그 사이면 선형적으로 증가하는 가중치 반환
def get_blend_weight(pred, lower=0.15, upper=0.25, max_weight=0.5):
    # 0 ~ 1 사이로 정규화된 위치 계산
    position = np.clip((pred - lower) / (upper - lower), 0, 1)
    
    # 가중치 계산
    return position * max_weight

if __name__ == '__main__':
    # Data Load
    raw_meta  = load_data('match_info.csv') # ./open_track/match_info.csv
    raw_test = load_data('test.csv') # ./open_track/test.csv
    sub      = load_data('sample_submission.csv') # ./open_track/sample_submission.csv

    # Drop & Merge
    meta = raw_meta.drop(columns=['season_id', 'competition_id', 'competition_name', 'country_name', 'season_name'])

    test_list = []
    for p in tqdm(raw_test['path']):
        test_list.append(pd.read_csv('./open_track1/'+p))
    
    test = pd.concat(test_list, axis=0, ignore_index=True)
    test = test.merge(meta, 'inner', on=['game_id'])

    # 학습때 저장한 통계 데이터 load
    mappings_path = './params/feature_maps.joblib'
    mappings = joblib.load(mappings_path)

    player_map = mappings['player_map']
    team_map   = mappings['team_map']
    grid_map   = mappings['grid_map']
    global_map = mappings['global_map']
    point      = mappings['point']
    type_med   = mappings['type_med']

    params_path = './params/best_params.yaml'

    with open(params_path, 'r') as f:
        params = yaml.safe_load(f)

    # feature 생성
    test_data = create_feature(test, player_map, team_map, grid_map, global_map, point, type_med, is_train=False)

    # object type의 columns를 category로 변경
    obj_cols = [i for i in test_data.columns if test_data[i].dtype == 'object']
    test_data[obj_cols] = test_data[obj_cols].astype('category')

    # columns drop
    test_drop_dx = drop_columns('dx')
    test_drop_dy = drop_columns('dy')

    # 모델 저장 path
    save_path_dx_mae = 'autogluon_models/model_dx_mae'
    save_path_dy_mae = 'autogluon_models/model_dy_mae'
    save_path_dx_quantile = 'autogluon_models/model_dx_quantile'
    save_path_dy_quantile = 'autogluon_models/model_dy_quantile'

    # model load
    predictor_dx_mae = TabularPredictor.load(save_path_dx_mae)
    predictor_dy_mae = TabularPredictor.load(save_path_dy_mae)
    predictor_dx_quantile = TabularPredictor.load(save_path_dx_quantile)
    predictor_dy_quantile = TabularPredictor.load(save_path_dy_quantile)
    
    print('inference..')
    # inference
    pred_dx_df = predictor_dx_quantile.predict(test_data.drop(columns=test_drop_dx))
    pred_dy_df = predictor_dy_quantile.predict(test_data.drop(columns=test_drop_dy))
    pred_dx_mae = predictor_dx_mae.predict(test_data.drop(columns=test_drop_dx))
    pred_dy_mae = predictor_dy_mae.predict(test_data.drop(columns=test_drop_dy))

    pred_dx_m = pred_dx_df[0.50]
    pred_dx_l = pred_dx_df[0.20]
    pred_dx_h = pred_dx_df[0.80]

    pred_dx = pred_dx_m.copy()

    xwl = get_blend_weight(pred_dx_m, lower=params['x_low_l'], upper=params['x_low_u'], max_weight=params['x_low_max_w'])
    pred_dx = (pred_dx * (1 - xwl) + (pred_dx_l * xwl))

    xwh = get_blend_weight(pred_dx_m, lower=params['x_high_l'], upper=params['x_high_u'], max_weight=params['x_high_max_w'])
    pred_dx = (pred_dx * (1 - xwh)) + (pred_dx_h * xwh)

    pred_dy_m = pred_dy_df[0.50]
    pred_dy_l = pred_dy_df[0.20]
    pred_dy_h = pred_dy_df[0.80]

    pred_dy = pred_dy_m.copy()

    ywl = get_blend_weight(pred_dy_m, lower=params['y_low_l'], upper=params['y_low_u'], max_weight=params['y_low_max_w'])
    pred_dy = (pred_dy * (1 - ywl) + (pred_dy_l * ywl))

    ywh = get_blend_weight(pred_dy_m, lower=params['y_high_l'], upper=params['y_high_u'], max_weight=params['y_high_max_w'])
    pred_dy = (pred_dy * (1 - ywh) + (pred_dy_h * ywh))

    ratio = params['ratio']
    pred_dx = (pred_dx * ratio) + (pred_dx_mae * (1-ratio))
    pred_dy = (pred_dy * ratio) + (pred_dy_mae * (1-ratio))

    # 절대 좌표로 복원
    pred_end_x = (test_data['last_start_x'] + pred_dx) * 105
    pred_end_y = (test_data['last_start_y'] + pred_dy) * 68

    # 경기장 범위 밖으로 나간 예측값 보정
    pred_end_x = pred_end_x.clip(0, 105)
    pred_end_y = pred_end_y.clip(0, 68)

    # pred df 생성
    pred_df = pd.DataFrame({
        'game_episode': test_data['game_episode'],
        'end_x': pred_end_x,
        'end_y': pred_end_y
    })
    # submission과 merge
    sub = sub[['game_episode']].merge(pred_df, on='game_episode', how='left')

    # 저장
    sub.to_csv('Autogluon_with_quantile_mae.csv', index=False)
    print('done.')