import pandas as pd
import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler,OneHotEncoder

def perform_clustering(sample_df, features):
    sample_df['player_dist_total'] = sample_df['player_dist_ride'] + sample_df['player_dist_walk']
    # sample_df['player_dist_total'] = np.log1p(sample_df['player_dist_total'])
    sample_df['drive_type'] = sample_df['player_dist_ride'].apply(lambda x: 
                                                                      0 if x == 0 else 1)
    sample_df['player_dmg'] = np.log1p(sample_df['player_dmg'])
    sample_df = sample_df[~(sample_df.isna().any(axis=1))]
    print("start-clustering")
    # 전체 데이터에 대한 클러스터링
    scaler = StandardScaler()
    all_features = scaler.fit_transform(sample_df[features])
    kmeans_all = KMeans(n_clusters=3, random_state=42)
    sample_df['cluster'] = kmeans_all.fit_predict(all_features)
    
    print("end-clustering and start onehot encoding ...")

    encoder = OneHotEncoder()
    cluster_encoded = encoder.fit_transform(sample_df[['cluster']]).toarray()
    cluster_encoded_df = pd.DataFrame(cluster_encoded, columns=[f'cluster_{i}' for i in range(cluster_encoded.shape[1])])

    # 원핫 인코딩 결과를 sample_df에 추가
    sample_df = sample_df.reset_index(drop=True)
    sample_df = pd.concat([sample_df, cluster_encoded_df], axis=1)

    print("end onehot encoding")

    return sample_df


if __name__ == "__main__":
    PROCESSED_DIR = "./data/processed/"
    MAP_TYPE = 'ERANGEL'
    SQUAD_TYPE = 2
    CLUSTERING_DIR = "./data/clustered/"
    os.makedirs(CLUSTERING_DIR, exist_ok=True)

    data_path = os.path.join(PROCESSED_DIR, "processed_kill_data.csv")
    clustered_data_path = os.path.join(CLUSTERING_DIR, "{}_{}_clustered_data.csv" .format(MAP_TYPE, SQUAD_TYPE))

    print("Loading processed data...")
    data = pd.read_csv(data_path)

    data_df = data[(data['map'] == MAP_TYPE) & (data['party_size'] == SQUAD_TYPE)]

    print("Performing clustering...")
    features = ['player_dist_total', 'player_dmg', 'drive_type', 'player_kills']
    
    clustered_data = perform_clustering(data_df, features)

    print("Saving clustered data...")
    clustered_data.to_csv(clustered_data_path, index=False)
    print(f"Clustered data saved to {clustered_data_path}")
