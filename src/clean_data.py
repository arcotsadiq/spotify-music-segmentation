
import pandas as pd

# Load the original dataset
df = pd.read_csv("data/dataset.csv")

print("Original dataset shape:", df.shape)

# Remove the unnecessary index column
df = df.drop(columns=["Unnamed: 0"], errors="ignore")

# Remove rows missing essential song information
df = df.dropna(subset=["track_id", "track_name", "artists"])

# Remove tracks with invalid duration or tempo
df = df[(df["duration_ms"] > 0) & (df["tempo"] > 0)]

# Remove exact duplicate rows only
df = df.drop_duplicates()

# Save the cleaned dataset
df.to_csv("data/cleaned_dataset.csv", index=False)

print("Cleaned dataset shape:", df.shape)
print("\nRemaining missing values:")
print(df.isnull().sum().sum())

print("\nCleaned dataset saved to data/cleaned_dataset.csv")
