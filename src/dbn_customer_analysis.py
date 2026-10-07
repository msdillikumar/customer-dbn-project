from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.neural_network import BernoulliRBM
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


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


def analyze_representation(hidden_2):
    """
    Analyze and display statistics of the learned 2D hidden representation.
    """
    h_df = pd.DataFrame(hidden_2, columns=["hidden_dim_1", "hidden_dim_2"])
    print("Hidden Representation Summary:")
    print(f"  Dimension 1: mean={h_df['hidden_dim_1'].mean():.4f}, std={h_df['hidden_dim_1'].std():.4f}, "
          f"min={h_df['hidden_dim_1'].min():.4f}, max={h_df['hidden_dim_1'].max():.4f}")
    print(f"  Dimension 2: mean={h_df['hidden_dim_2'].mean():.4f}, std={h_df['hidden_dim_2'].std():.4f}, "
          f"min={h_df['hidden_dim_2'].min():.4f}, max={h_df['hidden_dim_2'].max():.4f}")
    return h_df


def plot_hidden_representation(h_df, output_path="results/hidden_representation.png"):
    """
    Plot and save the 2D learned hidden representation using Matplotlib.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 5))
    plt.scatter(
        h_df["hidden_dim_1"],
        h_df["hidden_dim_2"],
        color="#2b5c8f",
        alpha=0.75,
        edgecolors="none",
        s=45
    )
    plt.title("Learned Customer Representation (DBN Hidden Space)", fontsize=12, pad=12)
    plt.xlabel("Hidden Dimension 1", fontsize=10)
    plt.ylabel("Hidden Dimension 2", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved hidden representation plot to {output_path}")


def cluster_representation(hidden_2, original_df, n_clusters=3, random_state=42):
    """
    Apply K-Means clustering to the learned 2D hidden representation.
    Calculates cluster statistics on original customer features.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(hidden_2)

    clustered_df = original_df.copy()
    clustered_df["cluster"] = labels

    print(f"\nDiscovered Clusters (k={n_clusters}):")
    cluster_counts = pd.Series(labels).value_counts().sort_index()
    for c_id, count in cluster_counts.items():
        print(f"  Cluster {c_id}: {count} customers ({count / len(labels) * 100:.1f}%)")

    feature_cols = [
        "website_visits",
        "purchase_frequency",
        "average_spend",
        "app_usage",
        "support_requests",
        "discount_usage"
    ]
    cluster_means = clustered_df.groupby("cluster")[feature_cols].mean().round(2)
    print("\nCluster Feature Averages (Original Feature Values):")
    print(cluster_means)

    return kmeans, labels, clustered_df, cluster_means


def plot_cluster_visualization(h_df, labels, kmeans, output_path="results/cluster_visualization.png"):
    """
    Plot and save K-Means clusters formed in the 2D DBN hidden space.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 5))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for c_id in np.unique(labels):
        mask = labels == c_id
        plt.scatter(
            h_df.loc[mask, "hidden_dim_1"],
            h_df.loc[mask, "hidden_dim_2"],
            label=f"Cluster {c_id}",
            color=colors[c_id % len(colors)],
            alpha=0.8,
            s=50,
            edgecolors="none"
        )

    # Plot centroids
    centers = kmeans.cluster_centers_
    plt.scatter(
        centers[:, 0],
        centers[:, 1],
        marker="X",
        s=130,
        color="red",
        edgecolors="black",
        linewidths=1.2,
        label="Centroids"
    )

    plt.title("Customer Behavior Clusters (DBN Hidden Space)", fontsize=12, pad=12)
    plt.xlabel("Hidden Dimension 1", fontsize=10)
    plt.ylabel("Hidden Dimension 2", fontsize=10)
    plt.legend(frameon=True)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved cluster visualization plot to {output_path}")


def save_results(h_df, labels, kmeans):
    """
    Save both hidden representation and cluster visualization plots.
    """
    plot_hidden_representation(h_df)
    plot_cluster_visualization(h_df, labels, kmeans)


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

    print("\n=== Step 4: Analyzing Hidden Representation ===")
    h_df = analyze_representation(hidden_2)

    print("\n=== Step 5: Applying K-Means to Hidden Representation ===")
    kmeans, labels, clustered_df, cluster_means = cluster_representation(hidden_2, df, n_clusters=3)

    print("\n=== Step 6: Saving Visualizations ===")
    save_results(h_df, labels, kmeans)


if __name__ == "__main__":
    main()
