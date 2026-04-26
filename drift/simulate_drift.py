import pandas as pd
import numpy as np
from pathlib import Path

DATA_PATH = "data/processed/uci_phishing_processed.csv"
OUTPUT_PATH = "drift/drifted_phishing_data.csv"

Path("drift").mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)
drifted_df = df.copy()

target_col = "Result"

np.random.seed(42)

drift_features = [
    "URL_Length",
    "having_Sub_Domain",
    "Prefix_Suffix",
    "SSLfinal_State",
    "Domain_registeration_length",
    "web_traffic",
    "Google_Index",
    "Links_pointing_to_page"
]

sample_idx = drifted_df.sample(frac=0.35, random_state=42).index

for feature in drift_features:
    if feature in drifted_df.columns:
        drifted_df.loc[sample_idx, feature] = -1

drifted_df.to_csv(OUTPUT_PATH, index=False)

print("Drifted dataset created successfully.")
print(f"Saved to: {OUTPUT_PATH}")
print(f"Rows: {drifted_df.shape[0]}")
print(f"Columns: {drifted_df.shape[1]}")
print("Drifted features:")
for feature in drift_features:
    if feature in drifted_df.columns:
        changed = (df[feature] != drifted_df[feature]).sum()
        print(f"- {feature}: {changed} changed values")