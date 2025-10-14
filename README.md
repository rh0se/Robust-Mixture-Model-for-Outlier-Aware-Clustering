# Robust Mixture Modeling for Outlier-Aware Clustering

## Overview
This project implements a **custom robust mixture model** based on the **Expectation–Maximization (EM)** algorithm to cluster synthetic Mars Rover ChemCam sensor data while explicitly accounting for outliers.

Standard clustering methods such as KMeans and Gaussian Mixture Models (GMMs) often fail when datasets contain systematic biases and noisy measurements. This project extends the traditional GMM framework by adding a **uniform component** to capture outliers, resulting in improved cluster purity and stability.

---

## Problem Motivation
Unsupervised clustering is commonly applied to scientific sensor data to reveal meaningful patterns without labeled examples. However, real-world data is often affected by:
- Measurement biases (e.g., silica concentration shifts at night)
- Random outliers (e.g., ChemCam misfires)
- Non-uniform noise patterns

Classical clustering methods assume well-behaved distributions and struggle in these conditions. To address this, we developed a **Gaussian–uniform mixture model** that improves robustness by **separating noise from structure**.

---

## Methodology

### 1. Synthetic Data Generation
- Three Gaussian clusters representing distinct rock classes  
- Nighttime silica readings were artificially biased  
- 5% uniformly distributed outliers were added to simulate sensor misfires

### 2. Bias Correction
Nighttime silica readings were corrected by subtracting a fixed offset to reduce systematic measurement shifts.

### 3. Model Architecture
We extend the GMM formulation by adding a **uniform distribution component** for outliers. The data likelihood under the mixture model is given by:

$$
p(x) = \sum_{k=1}^{K} \pi_k \, N\bigl(x \mid \mu_k, \Sigma_k\bigr) \;+\; \pi_{\mathrm{out}} \, U(x)
$$

where:  
- $x$ is a data point  
- $pi_k$ are the mixture weights for each Gaussian component  
- $N(x \mid \mu_k, \Sigma_k$ is the Gaussian probability density  
- $pi_{\mathrm{out}}$ is the weight for the outlier component  
- $U(x)$ is the uniform distribution over the data space




Here, **p(x)** represents the **overall probability density of the point x under the entire mixture model**, i.e. the sum of the contributions from all components (Gaussians + uniform). It is used during the E-step of EM to normalize responsibilities.

The EM algorithm alternates between:
- **E-Step**: Compute responsibilities for Gaussian and uniform components using the current parameters  
- **M-Step**: Update means, covariances, and mixture weights

### 4. Baselines
The proposed model was compared against:
- KMeans clustering  
- scikit-learn GaussianMixture model

---

## 📊 Results

| Method                   | Outlier Handling | Cluster Separation | Robustness to Noise |
|---------------------------|------------------|---------------------|-----------------------|
| KMeans                   |  No            | Poor               | Low                  |
| sklearn GMM              |  No            | Moderate           | Low                  |
| **Robust Mixture (Ours)** |  Yes           | Strong             | High                 |

PCA was used to reduce dimensionality and visualize cluster separation.  
The robust mixture model successfully isolated outliers into a separate component, unlike KMeans and standard GMM.

*(Insert PCA visualization figure here)*

---

## Key Takeaways
- Implemented EM algorithm for Gaussian–uniform mixtures **from scratch**  
- Improved clustering performance in the presence of noise and bias  
- Compared against standard methods to demonstrate clear benefits  
- Showcases both **algorithmic understanding** and **practical application**

---

## Tech Stack
- **Python**  
- NumPy, Matplotlib, Seaborn  
- scikit-learn (for baselines)  
- PCA for visualization

