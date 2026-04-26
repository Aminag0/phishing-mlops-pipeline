import pandas as pd
from scipy.io import arff
from pathlib import Path

RAW_DATA_PATH = Path("data/raw/Training Dataset.arff")
PROCESSED_DATA_PATH = Path("data/processed/uci_phishing_processed.csv")

def main():
    data, meta = arff.loadarff(RAW_DATA_PATH)
    df = pd.DataFrame(data)

    # Decode byte columns if any
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(lambda x: x.decode("utf-8") if isinstance(x, bytes) else x)

    print("Dataset loaded successfully")
    print("Shape:", df.shape)
    print("\nColumns:")
    print(df.columns.tolist())

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nClass distribution:")
    print(df.iloc[:, -1].value_counts())

    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"\nProcessed CSV saved to: {PROCESSED_DATA_PATH}")

if __name__ == "__main__":
    main()