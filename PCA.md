# Principal Component Analysis (PCA): A Rigorous Mathematical Exploration

Introduction

Principal Component Analysis (PCA) is one of the most widely used techniques for dimensionality reduction, feature extraction, and data compression in machine learning and statistics. PCA transforms data into a new coordinate system by identifying the directions (principal components) along which the data varies the most. This article delves deeply into the mathematical foundations of PCA, including its derivation, properties, and practical applications.

1. Mathematical Preliminaries and Intuition Behind PCA
PCA seeks to project data from a high-dimensional space onto a lower-dimensional subspace in such a way that the variance of the projected data is maximized. This is achieved by finding a new set of orthogonal axes (principal components) that capture the maximum variance in the data.

### 1.1 Notation and Setup
Given a dataset  
$X \in \mathbb{R}^{m \times n}$  with $m$ data points and $n$  features, let: $x_i \in \mathbb{R}^n$  denote the  $i$-th data point (a row of $X$).  
The goal is to find a transformation matrix  

$$
W \in \mathbb{R}^{n \times k}
$$  

(where $k \leq n$) that projects each  $x_i$  onto a lower-dimensional space.

### 1.2 Centering the Data

PCA requires centering the data around the mean. The mean vector  

$$
\mu \in \mathbb{R}^n
$$  

is defined as:

$$
\mu = \frac{1}{m} \sum_{i=1}^{m} x_i.
$$  

The centered data matrix $\tilde{X}$  

is obtained by subtracting the mean vector from each data point:

$$
\tilde{X} = X - \frac{1}{m} \mu^T,
$$  

where  $\frac{1}{m} \in \mathbb{R}^m$  

is a column vector of ones.

## Derivation of Principal Components

# Goal of PCA

The goal of PCA is to transform the data into a new coordinate system such that the greatest variance by any projection of the data lies on the first axis (the first principal component), the second greatest variance on the second axis, and so on.

## Example Setup

Consider a simple 2D dataset with three data points:

$$
X = \begin{bmatrix} 2 & 3 \\
3 & 4 \\
4 & 5 \end{bmatrix}.
$$

Each row represents a data point with two features.

### Step 1: Centering the Data

First, we compute the mean vector:

$$
\mu = \frac{1}{3} \sum_{i=1}^{3} x_i = \frac{1}{3} \left( \begin{bmatrix} 2 \\
3 \end{bmatrix} + \begin{bmatrix} 3 \\
4 \end{bmatrix} + \begin{bmatrix} 4 \\
5 \end{bmatrix} \right) = \begin{bmatrix} 3 \\
4 \end{bmatrix}.
$$

Now, we subtract the mean vector from each data point to obtain the centered data matrix:

$$
\tilde{X} = \begin{bmatrix} 2 - 3 & 3 - 4 \\
3 - 3 & 4 - 4 \\
4 - 3 & 5 - 4 \end{bmatrix} = \begin{bmatrix} -1 & -1 \\
0 & 0 \\ 1 & 1 \end{bmatrix}.
$$

### Step 2: Computing the Covariance Matrix

The covariance matrix is given by:

$$
\Sigma = \frac{1}{m} \tilde{X}^T \tilde{X},
$$

where \( m = 3 \) (number of data points). Compute \( \tilde{X}^T \tilde{X} \):

$$
\tilde{X}^T = \begin{bmatrix} -1 & 0 & 1 \\
-1 & 0 & 1 \end{bmatrix}, \quad \tilde{X}^T \tilde{X} = \begin{bmatrix} -1 & 0 & 1 \\
-1 & 0 & 1 \end{bmatrix} \begin{bmatrix} -1 & -1 \\
0 & 0 \\
1 & 1 \end{bmatrix} = \begin{bmatrix} 2 & 2 \\ 2 & 2 \end{bmatrix}.
$$

Now, the covariance matrix is:

$$
\Sigma = \frac{1}{3} \begin{bmatrix} 2 & 2 \\ 2 & 2 \end{bmatrix} = \begin{bmatrix} 0.67 & 0.67 \\
0.67 & 0.67 \end{bmatrix}.
$$

### Step 3: Eigenvalue Decomposition

To find the principal components, we perform the eigenvalue decomposition of the covariance matrix \( \Sigma \). We solve the characteristic equation:

$$
\text{det}(\Sigma - \lambda I) = 0,
$$

where \( I \) is the identity matrix. This gives:

$$
\text{det} \left( \begin{bmatrix} 0.67 - \lambda & 0.67 \\
0.67 & 0.67 - \lambda \end{bmatrix} \right) = 0.
$$

Expanding the determinant:

$$
(0.67 - \lambda)^2 - (0.67)^2 = 0.
$$

Simplifying:

$$
\lambda^2 - 1.34 \lambda = 0,
$$

which gives:

$$
\lambda(\lambda - 1.34) = 0 \implies \lambda_1 = 1.34, \, \lambda_2 = 0.
$$

The first eigenvalue corresponds to the direction of maximum variance.

### Step 4: Eigenvectors

To find the eigenvectors, we solve:

$$
(\Sigma - \lambda I)v = 0.
$$

For \( \lambda_1 = 1.34 \):

$$
\left( \begin{bmatrix} 0.67 & 0.67 \\
0.67 & 0.67 \end{bmatrix} - 1.34 \begin{bmatrix} 1 & 0 \\
0 & 1 \end{bmatrix} \right) v = \begin{bmatrix} -0.67 & 0.67 \\
0.67 & -0.67 \end{bmatrix} v = 0.
$$

The resulting eigenvector is \( v_1 = \begin{bmatrix} 1 \\ 1 \end{bmatrix} \) (after normalization). This is the first principal component.

## Singular Value Decomposition (SVD) and PCA

### Example Data

Consider the same centered data matrix:

$$
\tilde{X} = \begin{bmatrix} -1 & -1 \\
0 & 0 \\
1 & 1 \end{bmatrix}.
$$

### Step 1: Compute SVD

Perform the SVD of \( \tilde{X} \):

$$
\tilde{X} = U \Sigma V^T,
$$

where:
- \( U \) is an orthogonal matrix of left singular vectors.
- \( \Sigma \) is a diagonal matrix of singular values.
- \( V \) is an orthogonal matrix of right singular vectors.

Compute \( \tilde{X}^T \tilde{X} \) (already done):

$$
\tilde{X}^T \tilde{X} = \begin{bmatrix} 2 & 2 \\ 2 & 2 \end{bmatrix}.
$$

Eigenvalues are \( \lambda_1 = 1.34 \), \( \lambda_2 = 0 \), with corresponding eigenvectors \( v_1 = \begin{bmatrix} 1 \\ 1 \end{bmatrix} \) and \( v_2 = \begin{bmatrix} -1 \\ 1 \end{bmatrix} \).

### Step 2: Form Matrices

$$
V = \begin{bmatrix} \frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}} \\ \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \end{bmatrix}.
$$

Singular values \( \sigma_1 = 1.34 \approx 1.16 \), \( \sigma_2 = 0 \).

$$
\Sigma = \text{diag}(1.16, 0).
$$

## Maximizing Variance and Minimizing Reconstruction Error

### Maximizing Variance

Given a unit vector \( w \in \mathbb{R}^n \), PCA seeks to maximize:

$$
\text{Var}(\tilde{X} w) = w^T \Sigma w,
$$

subject to \( \|w\|^2 = 1 \). The solution corresponds to the eigenvector with the largest eigenvalue.

### Minimizing Reconstruction Error

PCA minimizes the reconstruction error:

$$
E = \|\tilde{X} - \tilde{X} WW^T\|_F^2,
$$

where \( W \) contains the top \( k \) eigenvectors. This provides the best low-rank approximation.

### Example of Reconstruction

Using the first principal component 
$$\( v_1 = \begin{bmatrix} 1 \\
1 \end{bmatrix} \)$$, project and reconstruct data:

$$
y_i = \tilde{x}_i \cdot v_1, \quad \text{reconstructed} = y_i v_1^T.
$$

example of python code

``` python
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Example dataset: Material properties (e.g., density, melting point, thermal conductivity)
data = {
    'Material': ['Material1', 'Material2', 'Material3', 'Material4', 'Material5'],
    'Density': [2.7, 7.8, 8.9, 2.3, 4.5],
    'Melting_Point': [660, 1538, 1455, 650, 1085],
    'Thermal_Conductivity': [237, 80, 401, 150, 235]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Extract features (excluding the Material column)
features = df.drop('Material', axis=1)

# Standardize the features
features_standardized = (features - features.mean()) / features.std()

# Apply PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(features_standardized)

# Create a DataFrame with the principal components
principal_df = pd.DataFrame(data=principal_components, columns=['Principal Component 1', 'Principal Component 2'])
principal_df['Material'] = df['Material']

# Plot the principal components
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Principal Component 1', y='Principal Component 2', hue='Material', data=principal_df, s=100)
plt.title('PCA of Material Properties')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.grid(True)
plt.show()

# Explained variance
explained_variance = pca.explained_variance_ratio_
print(f'Explained variance by Principal Component 1: {explained_variance[0]:.2f}')
print(f'Explained variance by Principal Component 2: {explained_variance[1]:.2f}')

# Principal component loadings
loadings = pca.components_
loading_df = pd.DataFrame(loadings.T, columns=['Principal Component 1', 'Principal Component 2'], index=features.columns)
print(loading_df)
```

# 5. Applications of PCA

## 5.1 Dimensionality Reduction

PCA reduces the number of features while retaining as much variance as possible, making it useful for:
- Preprocessing data
- Reducing noise
- Speeding up computation

## 5.2 Visualization

By projecting high-dimensional data onto two or three principal components, PCA enables visualization of complex datasets, making it easier to identify patterns and clusters.

## 5.3 Feature Extraction

PCA transforms correlated features into a smaller number of uncorrelated principal components, simplifying data interpretation and feature selection.

# 6. Practical Considerations

### Standardization

PCA is sensitive to the scale of the data, so features are often standardized (mean-centered and scaled to unit variance) before applying PCA. This ensures that all features contribute equally to the analysis.

### Choosing \( k \)

The number of principal components \( k \) is typically chosen based on the cumulative explained variance. A common approach is to select \( k \) such that it captures a threshold (e.g., 95%) of the total variance.
