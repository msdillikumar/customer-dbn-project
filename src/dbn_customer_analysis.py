from pathlib import Path
import pandas as pd
import numpy as np


def load_data(filepath="dataset/customer_data.csv"):
    """Load customer behavior dataset from CSV."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")
    data = pd.read_csv(path)
    return data


def preprocess_data(df):
    """
    Preprocess customer features into binary representation for Bernoulli RBM.
    Uses median thresholding: value >= median -> 1, value < median -> 0.
    """
    feature_cols = [
        "website_visits",
        "purchase_frequency",
        "average_spend",
        "app_usage",
        "support_requests",
        "discount_usage"
    ]
    features = df[feature_cols].copy()
    medians = features.median()
    binary_data = (features >= medians).astype(int)
    return binary_data.values, feature_cols, medians


def main():
    print("=== Step 1: Loading Dataset ===")
    df = load_data()
    print(f"Loaded {df.shape[0]} customer records with {df.shape[1]} features.")

    print("\n=== Step 2: Preprocessing Features ===")
    X_bin, feature_cols, medians = preprocess_data(df)
    print("Feature medians used for thresholding:")
    for col, med in medians.items():
        print(f"  {col}: {med:.2f}")

    print(f"\nPreprocessed binary shape: {X_bin.shape}")
    print("Sample preprocessed records (first 3):")
    print(X_bin[:3])


if __name__ == "__main__":
    main()
