# Deep Belief Network for Customer Behavior Representation

An academic Machine Learning project demonstrating unsupervised customer behavior representation learning using a small Deep Belief Network (DBN) constructed with stacked Bernoulli Restricted Boltzmann Machines (RBMs).

## Project Overview

- **Input**: 6 unlabeled customer behavior features
- **RBM Layer 1**: 6 visible units → 4 hidden units
- **RBM Layer 2**: 4 hidden units → 2 hidden units
- **Representation**: 2D hidden feature representation
- **Pattern Analysis**: K-Means clustering (k=3) for exploratory behavior pattern discovery

## Directory Structure

```
customer-dbn-project/
│
├── README.md
├── requirements.txt
│
├── dataset/
│   └── customer_data.csv
│
├── src/
│   └── dbn_customer_analysis.py
│
├── notebooks/
│   └── DBN_Analysis.ipynb
│
├── results/
│   ├── hidden_representation.png
│   └── cluster_visualization.png
│
└── screenshots/
    └── output.png
```
