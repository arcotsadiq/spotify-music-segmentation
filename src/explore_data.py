
import pandas as pd

# Load the Spotify dataset
file_path = "data/dataset.csv"
df = pd.read_csv(file_path)

# Display basic information
print("Dataset loaded successfully!")
print("\nFirst 5 songs:")
print(df.head())

print("\nDataset shape (rows, columns):")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nDataset information:")
df.info()

print("\nMissing values in each column:")
print(df.isnull().sum())

print("\nNumber of duplicate rows:")
print(df.duplicated().sum())

print("\nNumeric column statistics:")
print(df.describe())

print("\n--- Data Quality Checks ---")

print("Tracks with zero or negative duration:")
print((df["duration_ms"] <= 0).sum())

print("\nTracks with zero or negative tempo:")
print((df["tempo"] <= 0).sum())

print("\nUnique genre labels:")
print(df["track_genre"].nunique())

print("\nTracks per genre (first 10):")
print(df["track_genre"].value_counts().head(10))

print("\nMissing track IDs:")
print(df["track_id"].isnull().sum())

print("\nRepeated track IDs:")
print(df["track_id"].duplicated().sum())
