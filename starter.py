import pandas as pd

# Load the dataset
df = pd.read_csv("T100_Domestic_Market_and_Segment_Data_8942359590531559889.csv")

# Basic verification
print("Shape:", df.shape)
print("\nColumns:")
print(df.columns)

print("\nFirst 5 rows:")
print(df.head())

# Quick sanity checks
print("\nUnique origins:", df["origin"].nunique())
print("Unique destinations:", df["origin"].nunique())
print("Year range:", df["year"].min(), "to", df["year"].max())



