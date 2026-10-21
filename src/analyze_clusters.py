
import pandas as pd

# Load clustered song data
df = pd.read_csv("data/clustered_songs.csv")

# Audio features used by our clustering model
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

# Calculate the average audio features for each cluster
cluster_profiles = df.groupby("cluster")[features].mean()

print("Average audio-feature profile for each cluster:")
print(cluster_profiles.round(3).to_string())

# Inspect the most frequent genre labels in each cluster
print("\nMost frequent genre labels in each cluster:")

for cluster_id, group in df.groupby("cluster"):
    print(f"\nCluster {cluster_id}:")
    print(group["track_genre"].value_counts().head(5).to_string())

# Save the profiles for future analysis
cluster_profiles.to_csv("outputs/cluster_profiles.csv")

print("\nCluster profiles saved to outputs/cluster_profiles.csv")
