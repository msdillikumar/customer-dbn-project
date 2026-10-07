# Deep Belief Network for Customer Behavior Representation

An individual academic Machine Learning project demonstrating unsupervised customer behavior representation learning using a small Deep Belief Network (DBN) constructed by stacking two Bernoulli Restricted Boltzmann Machines (RBMs).

---

## 1. Problem Statement

A customer-analysis system contains unlabeled behavioral patterns spanning visits, transaction counts, spending, mobile engagement, support requests, and discount usage. Without ground-truth customer labels, traditional supervised classification cannot be applied. The problem is to develop a small, explainable Deep Belief Network that automatically learns compact, meaningful hidden representations from unlabeled customer behavior, and analyze the discovered patterns.

---

## 2. Objective

The primary objective is to build a small, simple, efficient, and explainable Deep Belief Network (DBN) that:
1. Ingests 6 unlabeled customer behavior features.
2. Converts continuous/discrete behavior indicators into binary representations suitable for Bernoulli RBMs.
3. Performs layer-wise unsupervised feature abstraction via stacked RBMs ($6 \rightarrow 4 \rightarrow 2$).
4. Extracts a 2-dimensional hidden representation for low-dimensional visualization.
5. Applies K-Means clustering ($k=3$) to discover and interpret customer behavioral groupings from the learned representations.

---

## 3. Concept

### Unsupervised Learning
Unsupervised learning models uncover underlying structures, probability distributions, or compact representations directly from input data without target labels or supervisory feedback.

### Restricted Boltzmann Machine (RBM)
A Restricted Boltzmann Machine is an energy-based, bipartite undirected graphical model consisting of:
- A visible layer $\mathbf{v}$ representing observable feature inputs.
- A hidden layer $\mathbf{h}$ capturing latent correlations.
- Symmetrical weight connections $\mathbf{W}$ between visible and hidden units, with no intra-layer connections (the "restricted" property).

For binary stochastic units, conditional probabilities are governed by sigmoid activations:
$$P(h_j = 1 \mid \mathbf{v}) = \sigma\left(b_j + \sum_i v_i W_{ij}\right)$$
$$P(v_i = 1 \mid \mathbf{h}) = \sigma\left(a_i + \sum_j h_j W_{ij}\right)$$

Training is conducted efficiently via **Contrastive Divergence (CD-1)**, approximating the gradient of log-likelihood through Gibbs sampling.

### Deep Belief Network (DBN)
A Deep Belief Network is a deep generative probabilistic model formed by stacking multiple RBMs hierarchically. It is trained using greedy layer-wise unsupervised learning:
- The first RBM is trained on the preprocessed input features.
- The hidden unit activation probabilities of the first RBM become the visible training inputs for the second RBM.
- Each successive layer learns higher-level, more abstract latent representations.

### Hidden Representation
The activations of the top-layer hidden units form a low-dimensional embedding (coordinate space) summarizing the essential variations in customer behavior.

---

## 4. Dataset

The project uses a synthetic customer behavior dataset (`dataset/customer_data.csv`) comprising 100 customer records across 6 behavioral attributes:

| Feature Name | Description | Value Type |
|---|---|---|
| `website_visits` | Monthly count of web portal visits | Integer |
| `purchase_frequency` | Number of completed orders per month | Integer |
| `average_spend` | Average transaction amount (in currency units) | Float |
| `app_usage` | Total monthly app session time / active hours | Float |
| `support_requests` | Number of customer support inquiries filed | Integer |
| `discount_usage` | Number of promotional vouchers or coupons redeemed | Integer |

> **Unlabeled Dataset**: The dataset contains strictly behavioral metrics without any target class, label, churn indicator, or predetermined segment columns.

---

## 5. Architecture

The network consists of two stacked Bernoulli RBM layers trained in a greedy, layer-wise fashion:

```
Customer Behavior Data (6 features)
               ↓
     Median Preprocessing
               ↓
   Visible Inputs: 6 binary units
               ↓
        [RBM Layer 1]
         6 visible → 4 hidden units
               ↓
   Hidden 1 Activations (4 units)
               ↓
        [RBM Layer 2]
         4 visible → 2 hidden units
               ↓
   Learned 2D Hidden Representation
               ↓
     K-Means Clustering (k=3)
               ↓
  Discovered Behavior Patterns
```

- **Input Dimension**: 6
- **RBM Layer 1**: 6 visible units $\rightarrow$ 4 hidden units (`n_components=4`)
- **RBM Layer 2**: 4 visible units $\rightarrow$ 2 hidden units (`n_components=2`)
- **Final Latent Dimension**: 2 continuous activation dimensions $[h_1, h_2]$

---

## 6. Methodology

1. **Dataset Loading**: Reads `dataset/customer_data.csv` using Pandas.
2. **Preprocessing**: Converts continuous/discrete behavior attributes into binary $\{0, 1\}$ vectors using feature-wise median thresholding.
3. **Layer 1 Greedy Pre-training**: Fits `BernoulliRBM` (6 visible $\rightarrow$ 4 hidden) on binary input vectors using Contrastive Divergence. Computes `hidden_1 = rbm1.transform(X_bin)`.
4. **Layer 2 Greedy Pre-training**: Fits `BernoulliRBM` (4 visible $\rightarrow$ 2 hidden) on `hidden_1`. Computes `hidden_2 = rbm2.transform(hidden_1)`.
5. **Latent Space Analysis**: Computes summary statistics (mean, standard deviation, bounds) of the resulting 2D representation and generates a 2D scatter plot (`results/hidden_representation.png`).
6. **Exploratory Clustering**: Applies K-Means ($k=3$) directly to the 2D hidden representation coordinates.
7. **Cluster Profiling**: Calculates the mean feature values of original customer behaviors within each cluster to interpret discovered patterns. Saves `results/cluster_visualization.png`.

---

## 7. Preprocessing

Bernoulli RBMs model binary stochastic visible units. To convert diverse numeric customer behaviors into binary inputs without complex transformation pipelines, we apply simple and explainable **median thresholding**:

$$\text{binary\_feature}_i = \begin{cases} 1 & \text{if } x_i \ge \text{median}(x_i) \\ 0 & \text{if } x_i < \text{median}(x_i) \end{cases}$$

### Feature Medians Computed on Dataset:
- `website_visits`: **16.00**
- `purchase_frequency`: **3.50**
- `average_spend`: **49.59**
- `app_usage`: **19.60**
- `support_requests`: **2.00**
- `discount_usage`: **4.00**

This method is robust to outliers, preserves relative behavioral rank, and is straightforward to explain during viva.

---

## 8. Results

### Learned 2D Hidden Representation Statistics

| Dimension | Mean | Std Dev | Min | Max |
|---|---|---|---|---|
| Hidden Dimension 1 | 0.4869 | 0.0189 | 0.4595 | 0.5104 |
| Hidden Dimension 2 | 0.4866 | 0.0186 | 0.4596 | 0.5099 |

- Output visualization saved to: `results/hidden_representation.png`

### Discovered Behavior Clusters ($k=3$)

| Cluster | Customer Count | Percentage | Key Behavioral Pattern |
|---|---|---|---|
| **Cluster 0** | 35 | 35.0% | High Digital Engagement & Frequent Buyers |
| **Cluster 1** | 41 | 41.0% | Low Engagement & High Support Inquiries |
| **Cluster 2** | 24 | 24.0% | Moderate Engagement & High Discount Usage |

### Mean Behavioral Values by Discovered Cluster

| Cluster | Website Visits | Purchase Freq | Avg Spend ($) | App Usage (hrs) | Support Requests | Discount Usage |
|---|---|---|---|---|---|---|
| **Cluster 0** | **27.43** | **12.37** | **$122.68** | **45.47** | 1.00 | 2.91 |
| **Cluster 1** | 10.17 | 2.12 | $49.53 | 9.38 | **4.27** | 4.12 |
| **Cluster 2** | 15.50 | 3.46 | $35.79 | 17.58 | 1.25 | **8.29** |

- Output cluster plot saved to: `results/cluster_visualization.png`

---

## 9. How to Run

### Prerequisites
- Python 3.10+ (tested on Python 3.13)
- pip

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run the Analysis Pipeline
```bash
python src/dbn_customer_analysis.py
```

### Step 3: Launch Jupyter Notebook (Optional)
To walk through the interactive analysis step-by-step:
```bash
jupyter notebook notebooks/DBN_Analysis.ipynb
```

---

## 10. Folder Structure

```
customer-dbn-project/
│
├── README.md                          # Project documentation
├── requirements.txt                   # Core dependencies
│
├── dataset/
│   └── customer_data.csv              # Unlabeled synthetic customer dataset (100 rows)
│
├── src/
│   └── dbn_customer_analysis.py       # Modular Python implementation
│
├── notebooks/
│   └── DBN_Analysis.ipynb             # Step-by-step interactive walkthrough
│
├── results/
│   ├── hidden_representation.png      # 2D DBN latent space plot
│   └── cluster_visualization.png      # 2D latent space with K-Means clusters
│
└── screenshots/
    └── output.png                     # Verified terminal execution output
```

---

## 11. Technologies

- **Python** (Core language)
- **NumPy** (Numerical arrays and seed management)
- **Pandas** (Data loading, median transformation, and statistical aggregation)
- **Scikit-learn** (`sklearn.neural_network.BernoulliRBM`, `sklearn.cluster.KMeans`)
- **Matplotlib** (Scientific 2D scatter plots)
- **Jupyter Notebook** (Demonstration and interactive viva walkthrough)

---

## 12. Limitations

1. **Synthetic Dataset**: Built on 100 generated customer records for clear educational demonstration rather than massive production enterprise streams.
2. **Absence of Ground Truth**: In pure unsupervised learning, cluster labels represent discovered natural groupings rather than validated external truth.
3. **Binary Input Assumption**: Bernoulli RBM assumes binary inputs; continuous behaviors are binarized via median thresholding, discarding intra-bin granularity.
4. **Small DBN Scale**: The two-layer ($6 \rightarrow 4 \rightarrow 2$) network is tailored for explainability, fast execution, and direct 2D projection, not for high-capacity generative modeling.
5. **Sensitivity to Initialization**: RBM training and K-Means clustering outcomes depend on weight initialization and random seed selection.

---

## 13. Conclusion

The small Deep Belief Network successfully demonstrated unsupervised representation learning on unlabeled customer behavior data. By stacking two Bernoulli Restricted Boltzmann Machines ($6 \rightarrow 4 \rightarrow 2$), the model compressed 6 behavioral indicators into a coherent 2-dimensional hidden representation. Subsequent K-Means clustering ($k=3$) in this latent space identified three distinct customer profiles: high-value digital shoppers, low-engagement support-reliant users, and discount-driven shoppers. The project confirms the utility of stacked RBMs for compact latent feature discovery in an efficient, viva-friendly framework.

---

## 14. References & Acknowledgements

- Hinton, G. E., Osindero, S., & Teh, Y. W. (2006). *A fast learning algorithm for deep belief nets*. Neural Computation, 18(7), 1527-1554.
- Hinton, G. E. (2012). *A practical guide to training restricted Boltzmann machines*. Neural Networks: Tricks of the Trade, 599-619.
- Scikit-Learn Documentation: `sklearn.neural_network.BernoulliRBM` & `sklearn.cluster.KMeans`.
