### The k-Nearest Neighbors (k-NN) algorithm 
It is one of the most well-known algorithms in machine learning due to its simplicity and versatility. However, despite its apparent ease of implementation, a more rigorous understanding reveals deeper implications regarding its mathematical underpinnings, computational complexity, and decision boundaries. 

This section delves into the k-NN algorithm with technical precision, discussing:

- Its theoretical basis,
- Its relation to other machine learning methods,
- Its strengths and limitations from a statistical learning perspective.

- ## What is k-Nearest Neighbors?

k-Nearest Neighbors (k-NN) is a **lazy learning algorithm**, meaning it does not build an explicit model during training. Instead, it stores the entire training dataset and makes decisions based on the *k* closest data points when a new input is given. The algorithm works by:

1. Calculating the distance between the query point and the training points.
2. Identifying the nearest neighbors.
3. Making a prediction based on the majority label (for classification) or the average value (for regression).

### Steps of k-NN:

1. **Store the training data**: There is no explicit training phase beyond storing the dataset.
2. **Compute the distance**: When a query point is provided, compute its distance to all the training points.
3. **Identify the nearest neighbors**: Sort the distances and select the *k* closest points.
4. **Make a prediction**:
   - For **classification**, return the majority label of the *k* neighbors.
   - For **regression**, return the average of the neighbors' values.
- 
### 1. Lazy Learning and Non-parametric Nature

One of the key characteristics of k-NN is that it is a **lazy learning algorithm**, meaning that it defers the work of generalization until query time, rather than forming an explicit model during training (like most parametric algorithms such as linear regression or decision trees). This **non-parametric nature** is critical for understanding k-NN because:

- **Non-parametric algorithms** make no strong assumptions about the underlying data distribution. In contrast to parametric methods (which assume, for example, that data is normally distributed), k-NN makes no assumptions about the data’s functional form. This makes k-NN highly flexible but also prone to overfitting if $k$ is chosen too small.
  
- **Lazy learning** implies that all computational complexity is deferred to prediction time, rather than during training. In practical terms, this means that k-NN is computationally trivial during the "training" phase, where it simply stores the dataset. However, at prediction time, it has to calculate distances between the query point and all training points, which results in a time complexity of:

  $$
  O(N \cdot n)
  $$

  where $N$ is the size of the training set and $n$ is the number of features.

- The lack of a model-building step distinguishes k-NN from **eager learning methods**, where a model is explicitly constructed during the training phase.

# Mathematical Derivation and Formulation

The k-Nearest Neighbors (k-NN) algorithm can be viewed as a geometric approach to classification and regression, where predictions are based on distances between points in a vector space. To formalize this rigorously, we first need to define the space, the distance metrics, and the method for prediction.

## 1. Distance Metric

The distance metric is central to the k-NN algorithm as it defines "nearness" in the context of the feature space. Let the feature space be an $n$-dimensional real vector space $\mathbb{R}^n$, where each data point is a vector:

$$
\mathbf{x} = (x_1, x_2, \dots, x_n) \in \mathbb{R}^n
$$

For any two points $\mathbf{x}, \mathbf{y} \in \mathbb{R}^n$, we need to measure their distance. The most common distance functions used are:

### Euclidean Distance

The Euclidean distance between two points $\mathbf{x}$ and $\mathbf{y}$ is the straight-line distance between them, given by the following formula:

$$
d(\mathbf{x}, \mathbf{y}) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}
$$

This is the standard $\ell_2$-norm of the difference between the two vectors $\mathbf{x}$ and $\mathbf{y}$. In mathematical terms:

$$
d(\mathbf{x}, \mathbf{y}) = \|\mathbf{x} - \mathbf{y}\|_2
$$

Where $\|\mathbf{z}\|_2$ is the $\ell_2$-norm of the vector $\mathbf{z}$.

### Manhattan Distance

The Manhattan distance measures the distance along axes at right angles and is the sum of the absolute differences of the coordinates:

$$
d(\mathbf{x}, \mathbf{y}) = \sum_{i=1}^{n} |x_i - y_i|
$$

This is the $\ell_1$-norm of the vector difference:

$$
d(\mathbf{x}, \mathbf{y}) = \|\mathbf{x} - \mathbf{y}\|_1
$$

### Minkowski Distance

The Minkowski distance is a generalization of both the Euclidean and Manhattan distances, parameterized by a value $p$:

$$
d(\mathbf{x}, \mathbf{y}) = \left( \sum_{i=1}^{n} |x_i - y_i|^p \right)^{\frac{1}{p}}
$$

For $p = 2$, we recover the Euclidean distance, and for $p = 1$, we recover the Manhattan distance. The Minkowski distance is the $\ell_p$-norm of the vector difference:

$$
d(\mathbf{x}, \mathbf{y}) = \|\mathbf{x} - \mathbf{y}\|_p
$$

## 2. Nearest Neighbors Search

Given a dataset $D = \{(\mathbf{x}_1, y_1), (\mathbf{x}_2, y_2), \dots, (\mathbf{x}_N, y_N)\}$, where $\mathbf{x}_i \in \mathbb{R}^n$ is the feature vector of the $i$-th point and $y_i \in \mathbb{R}$ or $y_i \in \{1, \dots, C\}$ is its corresponding label, our goal is to predict the label $\hat{y}$ of a new input point $\mathbf{q} \in \mathbb{R}^n$.

The process is as follows:

1. **Compute the distances** between the query point $\mathbf{q}$ and all training points $\mathbf{x}_i$, using the chosen distance metric $d(\mathbf{q}, \mathbf{x}_i)$.
2. **Sort the distances** in increasing order and select the indices of the $k$ closest points.
3. Let $N_k(\mathbf{q}) \subset D$ be the subset of the $k$ nearest neighbors to $\mathbf{q}$:

$$
N_k(\mathbf{q}) = \{(\mathbf{x}_{i_1}, y_{i_1}), (\mathbf{x}_{i_2}, y_{i_2}), \dots, (\mathbf{x}_{i_k}, y_{i_k})\}
$$

Where the indices $i_1, i_2, \dots, i_k$ are such that:

$$
d(\mathbf{q}, \mathbf{x}_{i_j}) \leq d(\mathbf{q}, \mathbf{x}_{i_{j+1}}) \quad \text{for all} \ j = 1, \dots, k-1
$$

## 3. Classification

For classification tasks, the prediction is based on a **majority voting rule**. Let $y_{i_j} \in \{1, 2, \dots, C\}$ be the label of the $j$-th nearest neighbor in $N_k(\mathbf{q})$, where $C$ is the number of possible classes. The predicted label $\hat{y}$ for the query point $\mathbf{q}$ is the most frequent class label among the $k$-nearest neighbors:

$$
\hat{y} = \arg\max_{c \in \{1, \dots, C\}} \sum_{j=1}^{k} 1(y_{i_j} = c)
$$

Here, $1(\cdot)$ is the **indicator function**, which returns 1 if the argument is true and 0 otherwise.

In the case of **binary classification** (where $C = 2$, say $y \in \{0, 1\}$), this simplifies to:

$$
\hat{y} = \begin{cases}
1 & \text{if} \ \sum_{j=1}^{k} y_{i_j} > \frac{k}{2} \\
0 & \text{otherwise}
\end{cases}
$$

This is equivalent to voting: if more than half of the $k$ neighbors belong to class 1, the prediction is 1, otherwise it's 0.

## 4. Regression

For regression tasks, the prediction is the **average** of the labels of the $k$-nearest neighbors. Let $y_{i_j} \in \mathbb{R}$ be the real-valued label of the $j$-th nearest neighbor in $N_k(\mathbf{q})$. The predicted value $\hat{y}$ for the query point $\mathbf{q}$ is the mean of the nearest neighbors' labels:

$$
\hat{y} = \frac{1}{k} \sum_{j=1}^{k} y_{i_j}
$$

This is simply the arithmetic average of the $k$-nearest neighbors' values.

## 5. Choosing the Value of $k$

The value of $k$, which represents the number of neighbors to consider, is a critical **hyperparameter**. If $k$ is too small, the model becomes sensitive to noise, while if $k$ is too large, the prediction may become overly smooth, missing important local patterns. Common heuristics for selecting $k$ include:

- **Cross-validation** to optimize $k$.
- Using the rule of thumb: $k = \sqrt{N}$, where $N$ is the number of training samples.

For large datasets, **approximate nearest neighbors search methods** (e.g., using KD-trees or Ball trees) can improve computational efficiency, as the brute-force approach of computing all pairwise distances has time complexity $O(N \cdot n)$, where $N$ is the number of training points and $n$ is the number of features.

## 2. Geometry and Decision Boundaries

From a geometric perspective, k-NN defines the label of a query point by evaluating its "local neighborhood." This makes k-NN a geometrically intuitive algorithm because its decision-making process relies on proximity, measured using distance metrics (e.g., Euclidean distance).

### Voronoi Diagrams and Decision Boundaries

The decision boundaries created by k-NN are piecewise linear and are influenced by the distribution of the training points. More formally, the decision boundary formed by k-NN with $k=1$ (1-NN) corresponds to a Voronoi diagram, where each region contains all points closest to a particular training sample. The boundaries between these regions are determined by the distance metric used.

In general, for $k > 1$, the decision boundaries become smoother since each region is now defined by the majority vote of the $k$ nearest neighbors rather than the nearest neighbor alone. This smoothing effect makes the boundaries less sensitive to noise but can also cause over-smoothing if $k$ is chosen too large.

### Curse of Dimensionality

While k-NN performs well in low-dimensional spaces, its performance can degrade in high-dimensional spaces due to the "curse of dimensionality." As the number of dimensions increases, the concept of "distance" becomes less informative. In high-dimensional spaces, the distance between points tends to become uniformly large. Mathematically, the volume of the feature space grows exponentially with the number of dimensions, and data points become sparsely distributed across this space. Thus, distance metrics like Euclidean distance lose their discriminative power, making it difficult to determine meaningful nearest neighbors.

One way to mitigate the curse of dimensionality is through **dimensionality reduction techniques** (e.g., Principal Component Analysis or t-SNE) before applying k-NN.

## 3. Statistical Properties and Bias-Variance Trade-off

A crucial aspect of understanding k-NN involves analyzing its **bias-variance trade-off**, a core concept in statistical learning theory.

- **Bias** refers to the error introduced by approximating a real-world problem with a simplified model. For k-NN, as $k$ increases, the bias typically increases because more neighbors are considered, making the model less flexible.
  
- **Variance** refers to the error introduced by the model’s sensitivity to small fluctuations in the training set. For small values of $k$, the variance is high because the algorithm is sensitive to noise in the training data.

This trade-off is controlled by $k$:
- For small $k$, k-NN has **low bias** (can model complex decision boundaries) but **high variance** (sensitive to noise).
- For large $k$, k-NN has **high bias** (smoother decision boundaries) but **low variance** (less sensitivity to individual data points).

Selecting the optimal value of $k$ is critical for minimizing generalization error, and this is typically done via cross-validation.

## 4. Weighted k-NN

In the standard k-NN algorithm, all neighbors are treated equally. However, in **weighted k-NN**, closer neighbors are given more weight in the decision-making process. Mathematically, we assign a weight $w_i$ to each neighbor based on its distance to the query point $\mathbf{q}$.

One common choice of weighting is to use the inverse distance:

$$
w_i = \frac{1}{d(\mathbf{q}, \mathbf{x}_i)}
$$

Where $d(\mathbf{q}, \mathbf{x}_i)$ is the distance between the query point $\mathbf{q}$ and the $i$-th training point. The prediction for a classification task would then be based on a **weighted majority vote**, and for regression, it would be a **weighted average**:

$$
\hat{y} = \frac{\sum_{i=1}^{k} w_i y_i}{\sum_{i=1}^{k} w_i}
$$

Where $y_i$ is the label of the $i$-th nearest neighbor. Weighted k-NN is particularly useful when closer neighbors are more likely to be informative than distant ones.

## 5. Complexity Analysis

### Time Complexity

The computational complexity of k-NN primarily stems from the need to compute the distance between the query point and each training point. For a dataset with $N$ training points, each with $n$ features, computing the distance between the query point and all training points takes:

$$
O(N \cdot n)
$$

Sorting the distances to find the nearest neighbors adds an additional factor of $O(N \log N)$. Thus, the total time complexity of predicting the label for a single query point using brute-force k-NN is:

$$
O(N \cdot n + N \log N)
$$

For large datasets, this can become expensive, especially in high-dimensional feature spaces. To address this, various **approximate nearest neighbor search techniques** have been developed, such as:

- **KD-trees**: A space-partitioning data structure that reduces the complexity of the nearest neighbor search in low-dimensional spaces. The search complexity is theoretically $O(\log N)$, but its performance degrades in high dimensions.
- **Ball Trees**: Another spatial data structure optimized for k-NN queries, more effective in moderate to high-dimensional spaces.

### Space Complexity

k-NN must store the entire training set, resulting in a space complexity of:

$$
O(N \cdot n)
$$

This memory requirement can be a limitation when dealing with large datasets.

###  Detecting Defects in a 2D Lattice Using K-Nearest Neighbors (K-NN)
we will explore a Python program that simulates a 2D square lattice of atoms, introduces random defects (vacancies and displaced atoms), and then classifies the defects using a K-Nearest Neighbors (K-NN) classifier. We will walk through each step of the process, from generating the lattice to applying the machine learning model to detect defects. This simulation has applications in materials science, where detecting defects in a crystalline lattice can provide insights into material properties and behaviors.

## Step 1: Create a 2D Square Lattice of Atoms

The code first creates a perfect 2D square lattice where each atom is arranged in a grid with regular spacing between them.

### Parameters:
- **rows** and **cols**: These parameters specify the number of rows and columns in the grid.
- **spacing**: Defines the distance between adjacent atoms.
- `np.meshgrid`: Creates the grid of \(x\) and \(y\) coordinates, and `np.column_stack` combines them into a list of \((x, y)\) positions.

**Output**: A perfect lattice where every atom has a predefined position in a 2D space.

## Step 2: Introduce Defects (Vacancies and Displacements)

Defects are introduced into the lattice in two forms:

- **Vacancies**: Some atoms are randomly removed from the lattice.
- **Displacements**: A few atoms are shifted from their original position by a small random amount.
### Vacancies:
Random atoms are removed from the lattice using `np.random.choice`, which selects random indices to remove.

### Displacements:
A few atoms that are not vacancies are slightly displaced. The displacement is applied within a range controlled by `displacement_range`.

**Output**: A modified lattice with missing atoms (vacancies) and some atoms displaced from their original positions.

## Step 3: Label Defects

In this step, the code assigns labels to each atom in the lattice:

- **1** for defective atoms (either vacancies or displaced).
- **0** for non-defective atoms.

The function initializes a list of zeros (representing non-defective atoms).  
It then marks the indices corresponding to vacancies and displacements as **1** to represent defects.

**Output**: An array of binary labels where **1** indicates defective atoms, and **0** indicates non-defective atoms.

## Step 4: Compute Features for Each Lattice Point

Instead of directly using the positions of atoms, this step computes features based on the distances between atoms and their nearest neighbors. This is important for the K-NN classifier because it uses this information to identify patterns of defects.
### NearestNeighbors:
This computes the nearest neighbors for each atom using a \(k\) value to determine how many neighbors to consider.

- **k+1**: We use \(k+1\) because the closest neighbor will always be the point itself, and we exclude it by flattening the distances and ignoring the first distance.
- **features**: These are arrays of distances to the nearest neighbors, which will be used as input for the K-NN classifier.

**Output**: A set of features for each atom that captures its spatial relationship with its nearest neighbors.

## Step 5: K-NN Classification

The main part of the script involves training a K-NN classifier to detect defects based on the computed features.
### Training:
The data is split into training and test sets using `train_test_split`. The K-NN model is trained on the training set and then tested on the test set.

### Accuracy:
After training, the model’s accuracy is calculated by comparing its predictions to the actual labels.

**Output**: The K-NN classifier is trained to distinguish defective atoms from non-defective ones. The function returns the classification accuracy, predictions on the test set, and the test data.

## Step 6: Visualization

The code includes visualization functions to plot the lattice and show which atoms are defective.
### Lattice Plot:
The atoms are plotted on a 2D grid, color-coded by their labels (**red** for defective and **blue** for non-defective atoms).

### Scatter Plot:
A scatter plot is generated using `plt.scatter`, and atoms are shown as points.

Two visualizations are created:
1. The entire lattice with actual defects.
2. The test set with predicted defects, allowing for a comparison of the model’s predictions against the actual defects.


```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.model_selection import train_test_split

# Step 1: Create a perfect 2D square lattice of atoms
def create_lattice(rows, cols, spacing):
    x_positions = np.arange(0, cols * spacing, spacing)
    y_positions = np.arange(0, rows * spacing, spacing)
    X, Y = np.meshgrid(x_positions, y_positions)
    return np.column_stack((X.flatten(), Y.flatten()))

# Step 2: Introduce defects (vacancies and displaced atoms)
def introduce_defects(lattice, num_vacancies=5, num_displacements=5, displacement_range=0.05):
    lattice_with_defects = lattice.copy()
    
    # Introduce vacancies by randomly removing atoms
    vacancy_indices = np.random.choice(len(lattice), num_vacancies, replace=False)
    lattice_with_defects = np.delete(lattice_with_defects, vacancy_indices, axis=0)
    
    # Keep track of displaced atom indices
    remaining_indices = np.setdiff1d(np.arange(len(lattice)), vacancy_indices)
    displacement_indices = np.random.choice(remaining_indices, num_displacements, replace=False)
    
    # Randomly displace selected atoms
    displacements = np.random.uniform(-displacement_range, displacement_range, (num_displacements, 2))
    for idx, disp in zip(displacement_indices, displacements):
        idx_in_defects = idx - np.searchsorted(vacancy_indices, idx)
        lattice_with_defects[idx_in_defects] += disp
    
    return lattice_with_defects, vacancy_indices, displacement_indices

# Step 3: Label defects (vacancies and displaced atoms as 1, non-defects as 0)
def label_defects(lattice_size, vacancy_indices, displacement_indices):
    labels = np.zeros(lattice_size, dtype=int)
    labels[vacancy_indices] = 1  # Mark vacancies as defective
    labels[displacement_indices] = 1  # Mark displaced atoms as defective
    return labels

# Step 4: Compute features for each lattice point
def compute_features(lattice_with_defects, original_lattice, k=4):
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(lattice_with_defects)
    features = []
    for point in original_lattice:
        # Find distances to the nearest neighbors in the defective lattice
        distances, indices = nbrs.kneighbors([point])
        distances = distances.flatten()[1:]  # Exclude the point itself
        features.append(distances)
    return np.array(features)

# Step 5: K-NN classification to identify defects based on features
def classify_defects(features, labels, n_neighbors=5):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    
    # Predict on the test set and return accuracy
    accuracy = knn.score(X_test, y_test)
    y_pred = knn.predict(X_test)
    return accuracy, y_pred, X_test, y_test

# Step 6: Visualization function
def visualize_lattice(lattice, defective_labels, title):
    plt.figure(figsize=(8, 8))
    plt.scatter(lattice[:, 0], lattice[:, 1], c=defective_labels, cmap='coolwarm', s=50, edgecolor='k')
    plt.title(title)
    plt.xlabel('X position')
    plt.ylabel('Y position')
    plt.axis('equal')
    plt.show()

# Main part of the script
# Step 1: Generate a 2D lattice of atoms
rows, cols, spacing = 50, 50, 0.2
lattice = create_lattice(rows, cols, spacing)

# Step 2: Introduce defects (vacancies and displacements)
num_vacancies = 30
num_displacements = 30
lattice_with_defects, vacancy_indices, displacement_indices = introduce_defects(
    lattice, num_vacancies, num_displacements, displacement_range=0.05
)

# Step 3: Label defects (vacancies and displaced atoms are labeled as 1)
labels = label_defects(len(lattice), vacancy_indices, displacement_indices)

# Step 4: Compute features for each lattice point
features = compute_features(lattice_with_defects, lattice)

# Step 5: Use K-NN to classify defects
accuracy, y_pred, X_test, y_test = classify_defects(features, labels)

# Print classification accuracy
print(f'K-NN Classification Accuracy: {accuracy * 100:.2f}%')

# Step 6: Visualize the lattice with color-coded defective and non-defective atoms
visualize_lattice(lattice, labels, 'Lattice with Defects (Red = Defective, Blue = Non-Defective)')

# Optional: Visualize the test set classification results
# Map X_test back to lattice indices for visualization
test_indices = [np.where(np.all(features == x, axis=1))[0][0] for x in X_test]
predicted_labels = y_pred
visualize_lattice(lattice[test_indices], predicted_labels, 'K-NN Predicted Defects on Test Set')

```




```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap

# Generate synthetic phase data with noise for a more rigorous example
np.random.seed(42)

# Temperature and Pressure ranges
temperature = np.linspace(-100, 200, 1000)  # Temperature range from -100 to 200 degrees
pressure = np.linspace(0.1, 1000, 1000)     # Pressure range from 0.1 to 1000 units

# Generate grid of points for temperature and pressure
T, P = np.meshgrid(temperature, pressure)
T = T.flatten()
P = P.flatten()

# Phase classification: Create nonlinear boundaries for solid, liquid, gas phases
def classify_phase(T, P):
    if T < 0:
        return 0  # Solid
    elif 0 <= T <= 100 and P > 200:
        return 1  # Liquid
    else:
        return 2  # Gas

# Apply the phase classification with added noise to simulate experimental uncertainty
noise = np.random.normal(0, 10, T.shape)
phases = np.array([classify_phase(t + n, p + n) for t, p, n in zip(T, P, noise)])

# Create a feature matrix (Temperature, Pressure) and labels (Phase)
X = np.column_stack((T, P))  # Features: Temperature, Pressure
y = phases                   # Labels: Phase (0 = Solid, 1 = Liquid, 2 = Gas)

# Split the dataset into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the k-nearest neighbors classifier with k=5
knn = KNeighborsClassifier(n_neighbors=5)

# Fit the model using the training data
knn.fit(X_train, y_train)

# Predict the phase for the test set
y_pred = knn.predict(X_test)

# Calculate and print accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Create a fine grid for visualization of decision boundaries
T_grid, P_grid = np.meshgrid(np.linspace(-100, 200, 200), np.linspace(0.1, 1000, 200))
X_grid = np.column_stack((T_grid.flatten(), P_grid.flatten()))

# Predict the phase for each point in the grid
Z = knn.predict(X_grid)
Z = Z.reshape(T_grid.shape)

# Plotting the phase diagram and decision boundaries
plt.figure(figsize=(10, 6))
cmap = ListedColormap(['#FF9999', '#99FF99', '#9999FF'])

plt.contourf(T_grid, P_grid, Z, cmap=cmap, alpha=0.6)

# Scatter plot the training points
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cmap, s=20, edgecolor='k', label='Training Data')

# Add labels and titles
plt.colorbar()
plt.title('Phase Classification Using K-NN (Solid, Liquid, Gas)')
plt.xlabel('Temperature (°C)')
plt.ylabel('Pressure (atm)')
plt.legend(loc='upper left')
plt.show()

# Predict the phase of a new sample (e.g., T=50°C, P=300 atm)
new_sample = np.array([[50, 300]])
predicted_phase = knn.predict(new_sample)
print(f'Predicted phase for temperature=50°C and pressure=300 atm: {predicted_phase[0]}')
```





  
