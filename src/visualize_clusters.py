
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Load clustered songs and standardized features
df = pd.read_csv("data/clustered_songs.csv")
X = pd.read_csv("data/scaled_features.csv")

# Create output folder if it does not already exist
import os
os.makedirs("outputs", exist_ok=True)

# Chart 1: Number of songs in each cluster
cluster_counts = df["cluster"].value_counts().sort_index()

plt.figure(figsize=(8, 5))
cluster_counts.plot(kind="bar")
plt.title("Number of Songs in Each Cluster")
plt.xlabel("Cluster")
plt.ylabel("Number of Songs")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("outputs/cluster_sizes.png", dpi=300)
plt.close()

# Chart 2: PCA visualization on a reproducible sample
sample_size = min(5000, len(X))
sample_indices = X.sample(
    n=sample_size,
    random_state=42
).index

X_sample = X.loc[sample_indices]
labels = df.loc[sample_indices, "cluster"]

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_sample)

plt.figure(figsize=(9, 6))
scatter = plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=labels,
    cmap="viridis",
    s=10,
    alpha=0.6
)

plt.title("Spotify Song Clusters (PCA Visualization)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar(scatter, label="Cluster")
plt.tight_layout()
plt.savefig("outputs/cluster_visualization.png", dpi=300)
plt.close()

print("Visualizations created successfully!")
print("Saved: outputs/cluster_sizes.png")
print("Saved: outputs/cluster_visualization.png")
print(
    "Variance explained by two PCA components:",
    round(pca.explained_variance_ratio_.sum() * 100, 2),
    "%"
)
