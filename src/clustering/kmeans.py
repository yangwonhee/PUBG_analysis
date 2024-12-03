from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def perform_clustering(data, features, n_clusters=3):
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(data[features])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    data['cluster'] = kmeans.fit_predict(scaled_features)
    return data
