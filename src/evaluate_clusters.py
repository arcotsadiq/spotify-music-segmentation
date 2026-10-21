
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load the scaled audio features
X = pd.read_csv("data/scaled_features.csv")

# Test different numbers of clusters
cluster_counts = [2, 3, 4, 5, 6, 7, 8]
sample_size = min(5000, len(X))

print("Evaluating different numbers of clusters...\n")

for k in cluster_counts:
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X)

    score = silhouette_score(
        X,
        labels,
        sample_size=sample_size,
        random_state=42
    )

    print(f"Clusters: {k} | Silhouette Score: {score:.4f}")
