
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load cleaned data
df = pd.read_csv("data/cleaned_dataset.csv")

# Select audio features for clustering
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

X = df[features].copy()

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save the scaled features
scaled_df = pd.DataFrame(X_scaled, columns=features)
scaled_df.to_csv("data/scaled_features.csv", index=False)

print("Feature preparation successful!")
print("Original feature shape:", X.shape)
print("Scaled feature shape:", X_scaled.shape)
print("\nScaled feature means (approximately zero):")
print(scaled_df.mean().round(3))
print("\nScaled feature standard deviations (approximately one):")
print(scaled_df.std(ddof=0).round(3))
print("\nSaved to data/scaled_features.csv")
