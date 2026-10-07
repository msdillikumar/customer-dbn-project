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


def train_dbn(X, random_state=42):
    """
    Train a 2-layer Deep Belief Network:
      Layer 1: 6 visible inputs -> 4 hidden units
      Layer 2: 4 hidden inputs  -> 2 hidden units
    Returns trained RBM layers and learned representations.
    """
    # RBM Layer 1 (6 -> 4)
    rbm1 = BernoulliRBM(
        n_components=4,
        learning_rate=0.1,
        n_iter=50,
        batch_size=10,
        random_state=random_state,
        verbose=False
    )
    rbm1.fit(X)
    hidden_1 = rbm1.transform(X)

    # RBM Layer 2 (4 -> 2)
    rbm2 = BernoulliRBM(
        n_components=2,
        learning_rate=0.1,
        n_iter=50,
        batch_size=10,
        random_state=random_state,
        verbose=False
    )
    rbm2.fit(hidden_1)
    hidden_2 = rbm2.transform(hidden_1)

    return rbm1, rbm2, hidden_1, hidden_2


def main():
    print("=== Step 1: Loading Dataset ===")
    df = load_data()
    print(f"Loaded {df.shape[0]} customer records with {df.shape[1]} features.")

    print("\n=== Step 2: Preprocessing Features ===")
    X_bin, feature_cols, medians = preprocess_data(df)
    print("Feature medians used for thresholding:")
    for col, med in medians.items():
        print(f"  {col}: {med:.2f}")
    print(f"Preprocessed binary shape: {X_bin.shape}")

    print("\n=== Step 3: Training DBN (RBM 1: 6->4, RBM 2: 4->2) ===")
    rbm1, rbm2, hidden_1, hidden_2 = train_dbn(X_bin)
    print("DBN Layer 1 (6 -> 4) representation shape:", hidden_1.shape)
    print("DBN Layer 2 (4 -> 2) representation shape:", hidden_2.shape)
    print("\nSample learned 2D hidden representations (first 5 records):")
    for i in range(5):
        print(f"  Customer {i+1}: Dim 1 = {hidden_2[i, 0]:.4f}, Dim 2 = {hidden_2[i, 1]:.4f}")


if __name__ == "__main__":
    main()
