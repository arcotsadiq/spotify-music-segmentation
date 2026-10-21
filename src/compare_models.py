
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load the cleaned songs and scaled audio features
df = pd.read_csv("data/cleaned_dataset.csv")
X = pd.read_csv("data/scaled_features.csv")

# Audio features used for clustering
features = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
]

# Test the three models
cluster_counts = [2, 5, 7]
sample_size = min(5000, len(X))

for k in cluster_counts:
    print("\n" + "=" * 60)
    print(f"RESULTS FOR K = {k}")
    print("=" * 60)

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

    # Add temporary cluster labels; original files remain unchanged
    results = df.copy()
    results["cluster"] = labels

    print(f"\nSilhouette score: {score:.4f}")

    print("\nCluster sizes:")
    print(results["cluster"].value_counts().sort_index().to_string())

    print("\nAverage audio features per cluster:")
    profiles = results.groupby("cluster")[features].mean()
    print(profiles.round(3).to_string())

    print("\nTop 5 genres per cluster:")
    for cluster_id, group in results.groupby("cluster"):
        print(f"\nCluster {cluster_id}:")
        print(
            group["track_genre"]
            .value_counts()
            .head(5)
            .to_string()
        )

print("\nComparison complete. Original project files were not changed.")
