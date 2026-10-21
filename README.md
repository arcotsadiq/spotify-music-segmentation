# Spotify Music Genre Segmentation

## Overview
An unsupervised machine-learning project that groups Spotify tracks based on their audio characteristics using K-Means clustering.

## Objectives
- Explore and clean Spotify track data.
- Select and scale numerical audio features.
- Apply K-Means clustering to group similar tracks.
- Visualize clusters and analyze their audio characteristics.
- Compare clustering configurations using silhouette scores.

## Dataset
The project uses a Spotify tracks dataset containing track metadata, audio features, and genre labels.

The cleaned dataset contains **113,392 tracks**. Nine audio features are used for clustering:

- Danceability
- Energy
- Loudness
- Speechiness
- Acousticness
- Instrumentalness
- Liveness
- Valence
- Tempo

Genre labels are used to interpret the resulting clusters, not as training targets.

## Methodology
1. Explore the original dataset.
2. Clean invalid and incomplete records.
3. Standardize the selected audio features using `StandardScaler`.
4. Apply K-Means clustering.
5. Evaluate cluster separation using the silhouette score.
6. Visualize clusters using Principal Component Analysis (PCA).
7. Compare cluster profiles and genre distributions.

## Model Evaluation
The tested configurations produced these approximate silhouette scores:

| Number of clusters | Silhouette score |
|---:|---:|
| 2 | 0.2569 |
| 5 | 0.1754 |
| 7 | 0.1952 |

Five clusters are the current baseline for interpretability. Two clusters achieved the highest silhouette score among the tested configurations, so the final choice should consider both separation and usefulness.

## Visualizations
- `outputs/cluster_sizes.png` — number of tracks in each cluster.
- `outputs/cluster_visualization.png` — PCA visualization of the clusters.

## Project Structure
```text
spotify-music-segmentation/
├── data/
├── outputs/
├── src/
├── README.md
└── requirements.txt
```

## Technologies
- Python
- pandas
- NumPy
- scikit-learn
- Matplotlib
- Principal Component Analysis (PCA)
- K-Means clustering

## Limitations
The silhouette scores indicate overlapping clusters. Genre labels can be mixed within clusters because the model groups tracks by audio characteristics rather than directly classifying their genres.

## Future Improvements
- Compare additional clustering algorithms.
- Explore alternative feature selections.
- Tune the number of clusters.
- Build a simple interface for exploring clusters and similar tracks.
