from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def find_optimal_k(rfm_scaled, k_range=range(2, 10)):
    inertias = []
    silhouettes = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(rfm_scaled)
        inertias.append(kmeans.inertia_)
        silhouettes.append(silhouette_score(rfm_scaled, kmeans.labels_))
    return list(k_range), inertias, silhouettes

def train_kmeans(rfm_scaled, n_clusters=4):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(rfm_scaled)
    return kmeans, labels
