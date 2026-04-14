import pandas as pd

df = pd.read_csv("case_features.csv")

mean_duration = df["total_duration"].mean()
median_duration = df["total_duration"].median()
mode_duration = df["total_duration"].mode()[0]
print(f"Average case duration: {mean_duration:.2f} seconds")
print(f"Median case duration: {median_duration:.2f} seconds")
print(f"Most common case duration: {mode_duration:.2f} seconds")