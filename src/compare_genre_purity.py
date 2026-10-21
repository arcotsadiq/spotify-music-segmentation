
import pandas as pd
from sklearn.cluster import KMeans

df = pd.read_csv("data/cleaned_dataset.csv")
X = pd.read_csv("data/scaled_features.csv")

for k in [2, 5, 7]:
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )
    labels = model.fit_predict(X)

    results = df[["track_genre"]].copy()
    results["cluster"] = labels

    print(f"\n{'=' * 55}")
    print(f"GENRE CONCENTRATION: K = {k}")
    print("=" * 55)

    for cluster_id, group in results.groupby("cluster"):
        counts = group["track_genre"].value_counts()
        dominant_genre = counts.index[0]
        dominant_count = counts.iloc[0]
        total = len(group)
        percentage = dominant_count / total * 100

        print(
            f"Cluster {cluster_id}: "
            f"{total:,} songs | "
            f"Top genre: {dominant_genre} | "
            f"{percentage:.1f}% in top genre"
        )
