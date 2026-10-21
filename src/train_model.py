
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load the cleaned song data and scaled audio features
df = pd.read_csv("data/cleaned_dataset.csv")
X = pd.read_csv("data/scaled_features.csv")

# Create the K-Means clustering model
kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)

# Train the model and assign a cluster to every song
df["cluster"] = kmeans.fit_predict(X)

# Save songs with their cluster assignments
df.to_csv("data/clustered_songs.csv", index=False)

# Display the results
print("K-Means model trained successfully!")
print("\nNumber of songs:", len(df))
print("Number of clusters:", df["cluster"].nunique())

print("\nSongs assigned to each cluster:")
print(df["cluster"].value_counts().sort_index())

print("\nSample clustered songs:")
print(
    df[["track_name", "artists", "track_genre", "cluster"]]
    .head(10)
    .to_string(index=False)
)

# Evaluate clustering quality on a reproducible sample
sample_size = min(5000, len(X))
X_sample = X.sample(n=sample_size, random_state=42)
labels_sample = df.loc[X_sample.index, "cluster"]

score = silhouette_score(X_sample, labels_sample)

print("\nSample silhouette score:", round(score, 4))
print("\nSaved results to data/clustered_songs.csv")
