import pandas as pd
import numpy as np
import os

def preprocess_kill_data(kill_data, agg_data):
    kill_data['kill_distance'] = np.sqrt(
        (kill_data['killer_position_x'] - kill_data['victim_position_x']) ** 2 +
        (kill_data['killer_position_y'] - kill_data['victim_position_y']) ** 2
    )
    # 맵 외부 포지션 삭제.
    positions = ['killer_position_x', 'killer_position_y', 'victim_position_x', 'victim_position_y']
    for pos in positions:
        kill_data = kill_data[(kill_data[pos] >= 0) & (kill_data[pos] <= 800000)]

    # 생존시간 0인 사람 삭제
    merged_df = kill_data.merge(agg_data[(agg_data['player_survive_time'] == 0.0) & (agg_data['player_survive_time'] > 2700.0)]
                                    ,left_on=['match_id', 'victim_name'], 
                                    right_on=['match_id', 'player_name'], how='left', indicator=True)
    to_delete = merged_df[merged_df['_merge'] == 'both'].index
    kill_data = kill_data.drop(to_delete)

    # killer_name 컬럼의 #unknown 삭제
    # print(kill_match_cut['killer_name'].value_counts().head())
    kill_data = kill_data[~(kill_data['killer_name'] == '#unknown')]

    # 안움직인 사람 제외
    agg_data = agg_data[~(agg_data['player_dist_walk'] == 0.0)]

    # 자살 삭제.
    kill_data = kill_data[~(kill_data['killer_name'] == kill_data['victim_name'])]
    agg_data = agg_data[~(agg_data['player_survive_time'] > 2700.0)]

    merged_data = pd.merge(kill_data, agg_data, 
                       left_on=['match_id', 'killer_name'], 
                       right_on=['match_id', 'player_name'], 
                       how='left')
    dropcol = ['date', 'game_size', 'match_mode', 'player_name']
    merged_data.drop(columns = dropcol, inplace = True)

    return merged_data

if __name__ == "__main__":
    RAW_DIR = "./data/raw/"
    PROCESSED_DIR = "./data/processed/"
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Example: Process 'kill_match_stats_final_0.csv'
    kill_data_path = os.path.join(RAW_DIR, "deaths/kill_match_stats_final_0.csv")
    match_stat_path = os.path.join(RAW_DIR, "aggregate/agg_match_stats_0.csv")
    processed_kill_data_path = os.path.join(PROCESSED_DIR, "processed_kill_data.csv")
    print(kill_data_path, match_stat_path)
    print("Loading raw data...")
    kill_data = pd.read_csv(kill_data_path)
    agg_data = pd.read_csv(match_stat_path)
    print("Processing data...")
    processed_kill_data = preprocess_kill_data(kill_data, agg_data)

    print("Saving processed data...")
    processed_kill_data.to_csv(processed_kill_data_path, index=False)
    print(f"Processed data saved to {processed_kill_data_path}")
