from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os

def perform_clustering(data, features, n_clusters=3):
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(data[features])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    data['cluster'] = kmeans.fit_predict(scaled_features)
    return data

if __name__ == "__main__":
    PROCESSED_DIR = "./data/processed/"
    CLUSTERED_DIR = "./data/processed/"
    os.makedirs(CLUSTERED_DIR, exist_ok=True)

    data_path = os.path.join(PROCESSED_DIR, "processed_kill_data.csv")
    clustered_data_path = os.path.join(CLUSTERED_DIR, "clustered_data.csv")

    print("Loading processed data...")
    data = pd.read_csv(data_path)

    print("Performing clustering...")
    features = ['player_dist_total', 'player_dmg', 'drive_type', 'player_kills']
    clustered_data = perform_clustering(data, features)

    print("Saving clustered data...")
    clustered_data.to_csv(clustered_data_path, index=False)
    print(f"Clustered data saved to {clustered_data_path}")
