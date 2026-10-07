from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.neural_network import BernoulliRBM


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


def train_rbm1(X, n_components=4, learning_rate=0.08, n_iter=25, random_state=42):
    """
    Train the first Bernoulli RBM layer:
    Maps 6 input binary features to 4 hidden representation units.
    """
    rbm1 = BernoulliRBM(
        n_components=n_components,
        learning_rate=learning_rate,
        n_iter=n_iter,
        random_state=random_state,
        verbose=False
    )
    rbm1.fit(X)
    hidden_1 = rbm1.transform(X)
    return rbm1, hidden_1


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

    print("\n=== Step 3: Training RBM Layer 1 (6 -> 4) ===")
    rbm1, hidden_1 = train_rbm1(X_bin)
    print("RBM Layer 1 trained successfully.")
    print(f"RBM 1 Hidden representation shape: {hidden_1.shape}")
    print(f"RBM 1 Components (weights) shape: {rbm1.components_.shape}")
    print("Sample RBM 1 hidden activations (first 3 records):")
    print(np.round(hidden_1[:3], 4))


if __name__ == "__main__":
    main()
