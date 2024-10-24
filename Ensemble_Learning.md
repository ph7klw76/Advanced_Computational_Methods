# Ensemble Learning

## Introduction to Ensemble Learning

Ensemble learning is a machine learning paradigm where multiple models, or "learners," are combined to improve the performance of predictive or classification tasks. By aggregating the outputs of several models, ensemble methods seek to reduce the errors of individual models, whether due to high variance, bias, or noise in the data. This approach has been successfully applied across various domains, including **physics**, where complex and noisy datasets make ensemble methods especially useful.

In this extended blog, we will explore the mathematical framework of ensemble learning, focusing on various ensemble techniques such as **weighted averaging**, **voting**, **bagging**, and **boosting**. We'll also delve into the widely-used **Random Forest** algorithm and its applications in physics research. Finally, we will provide examples of Python code applied in a physics context, demonstrating the use of ensemble learning.

### Ensemble Learning in Machine Learning

**Ensemble learning** is a robust machine learning technique that seeks to combine multiple models (also known as learners) to create a superior composite model. The intuition behind ensemble learning is that the ensemble benefits from the diversity of its individual models, potentially reducing the overall prediction error by leveraging the "wisdom of the crowd."

### Mathematical Representation

Mathematically, if we have a set of $M$ models:

$$
\{f_1(x), f_2(x), \dots, f_M(x)\}
$$

The goal is to combine these models to produce an aggregated model $\hat{f}(x)$ that outperforms each individual learner on a given predictive task.

### Ensemble Techniques

Depending on the specific ensemble technique, such as:
- **Bagging**: Reduces variance by averaging predictions over multiple models trained on different subsets of data.
- **Boosting**: Sequentially trains models to focus on correcting errors made by previous models, reducing bias.
- **Stacking**: Combines the predictions of several base models using a meta-model to improve accuracy.

The manner in which models are combined varies between these techniques.

### Bias-Variance Tradeoff and Ensemble Learning

The prediction error for any given learner can be decomposed into three components:
- **Bias**: Error due to overly simplistic models that cannot capture the underlying pattern.
- **Variance**: Error due to models being too sensitive to training data variations.
- **Irreducible noise**: Error that cannot be reduced, as it is inherent in the data.

Ensemble learning often outperforms single models because it balances the bias-variance tradeoff, leveraging the strengths of individual models and reducing overall prediction error.

### 1. Bias-Variance Decomposition

The **bias-variance decomposition** is a cornerstone concept in statistical learning theory. It provides insight into the sources of error for a model and explains why ensemble learning methods like **bagging** and **boosting** can improve performance.

The expected prediction error for a model $\hat{f}(x)$ on a data point $x$ with a true output $y$ is given by:

$$
\mathbb{E}[(y - \hat{f}(x))^2] = \left( \text{Bias}[\hat{f}(x)] \right)^2 + \text{Var}[\hat{f}(x)] + \sigma^2.
$$

### Explanation of Terms

- **$\mathbb{E}[(y - \hat{f}(x))^2]$**: The expected mean squared error of the model $\hat{f}(x)$, where the expectation $\mathbb{E}[\cdot]$ is taken over all possible training sets of a given size (considering the inherent randomness in training data).
  
- **Bias**: This term quantifies the systematic error due to the model's assumptions. It measures the difference between the expected prediction $\mathbb{E}[\hat{f}(x)]$ and the true function $f(x)$:

$$
\text{Bias}[\hat{f}(x)] = \mathbb{E}[\hat{f}(x)] - f(x).
$$

  High bias occurs when the model is too simple and makes overly strong assumptions, such as in linear models when the true relationship is highly nonlinear.

- **Variance**: This term captures the sensitivity of the model to variations in the training set. It measures how much the predictions vary across different datasets:

$$
\text{Var}[\hat{f}(x)] = \mathbb{E}[(\hat{f}(x) - \mathbb{E}[\hat{f}(x)])^2].
$$

  High variance occurs when a model is overly complex and fits the idiosyncrasies (noise) in the training data rather than capturing the underlying pattern.

- **$\sigma^2$**: This represents the irreducible error, which is the variance of the noise in the data. No matter how good the model is, this component of the error cannot be eliminated, as it represents randomness or noise in the relationship between $x$ and $y$.
### 2. Mathematical Formulation of Bagging

Bagging (Bootstrap Aggregating) aims to reduce the variance of an individual model by training multiple models on different bootstrapped samples of the data and then averaging their predictions.

For a dataset $D = \{(x_1, y_1), (x_2, y_2), \dots, (x_n, y_n)\}$, each base learner $f_i(x)$ is trained on a bootstrapped sample $D_i$, which is created by sampling $n$ examples from $D$ with replacement. This introduces diversity among the learners.

For regression, the final ensemble prediction is the average of the predictions from the individual models:

$$
\hat{f}_{\text{bagging}}(x) = \frac{1}{M} \sum_{i=1}^{M} f_i(x),
$$

where $f_i(x)$ is the prediction from the $i$-th learner, and $M$ is the number of learners.

For classification, majority voting is used, where each base classifier $f_i(x)$ returns a class label, and the predicted class is the one that receives the most votes:

$$
\hat{y}_{\text{bagging}} = \arg\max_{y \in C} \sum_{i=1}^{M} I(f_i(x) = y),
$$

where $C$ is the set of possible classes, and $I(\cdot)$ is the indicator function, which is 1 if $f_i(x) = y$ and 0 otherwise.

### Explanation of Terms

- **$I(f_i(x) = y)$**: The indicator function outputs 1 if the prediction of model $f_i(x)$ equals class $y$, otherwise it outputs 0. The majority vote selects the class with the highest number of votes.
- **$\hat{f}_{\text{bagging}}(x)$**: The ensemble model's final prediction, which is the average of all the individual model predictions.
### 3. Mathematical Formulation of Random Forests

Random Forests build on bagging by introducing another layer of randomness—at each split in each tree, a random subset of features is considered, rather than all features. This decorrelates the trees, further reducing variance.

For regression, the prediction is the average of the predictions of all decision trees:

$$
\hat{f}_{\text{RF}}(x) = \frac{1}{M} \sum_{i=1}^{M} T_i(x),
$$

where $T_i(x)$ is the prediction of the $i$-th decision tree.

For classification, majority voting is used:

$$
\hat{y}_{\text{RF}} = \arg\max_{y \in C} \sum_{i=1}^{M} I(T_i(x) = y).
$$

#### Explanation of Terms:
- **$T_i(x)$**: The prediction of the $i$-th decision tree, which has been trained on a bootstrapped sample of the data and splits nodes based on a random subset of the features.
- **$\hat{f}_{\text{RF}}(x)$**: The Random Forest’s final prediction, which is the average (in regression) or majority vote (in classification) of the decision trees.

### 4. Mathematical Formulation of Boosting

Boosting is a sequential ensemble technique where models are trained to focus on the mistakes of previous models. **AdaBoost** (Adaptive Boosting) is one of the most well-known boosting algorithms, where weak learners (typically decision stumps) are trained iteratively, with each learner correcting the errors of the previous ones.

#### Initialize Weights:
All examples are initially given equal weight:

$$
w_i(1) = \frac{1}{n},
$$

where $n$ is the number of training examples.

#### Train Weak Learner:
At each iteration $t$, a weak learner $f_t(x)$ is trained to minimize the weighted classification error:

$$
\epsilon_t = \sum_{i=1}^{n} w_i(t) I(f_t(x_i) \neq y_i),
$$

where $I(f_t(x_i) \neq y_i)$ is 1 if the weak learner misclassifies $x_i$, and 0 otherwise.

#### Update Weights:
The weak learner's influence is based on its performance. The weight for the learner is calculated as:

$$
\alpha_t = \frac{1}{2} \ln\left(\frac{1 - \epsilon_t}{\epsilon_t}\right),
$$

and the weights of the data points are updated as follows:

$$
w_i(t+1) = w_i(t) \exp(-\alpha_t y_i f_t(x_i)),
$$

where $y_i \in \{-1, 1\}$ is the true label.

#### Final Prediction:
The final model is a weighted sum of the weak learners:

$$
\hat{f}(x) = \text{sign}\left(\sum_{t=1}^{T} \alpha_t f_t(x)\right).
$$

#### Explanation of Terms:
- **$w_i(t)$**: The weight of the $i$-th training example at iteration $t$. Misclassified examples have their weights increased so that the next learner pays more attention to them.
- **$\epsilon_t$**: The error rate of the weak learner at iteration $t$.
- **$\alpha_t$**: The weight of the weak learner, which is larger for more accurate learners.
- **$f_t(x)$**: The $t$-th weak learner's prediction.

### 5. Variance Reduction via Bagging

A key benefit of bagging is variance reduction. If we assume that the base models are independent (a simplifying assumption), the variance of the ensemble is reduced as follows:

$$
\text{Var}\left(\hat{f}_{\text{bagging}}(x)\right) = \frac{1}{M} \text{Var}(f(x)).
$$

However, if the models are not independent (which is often the case), we must account for the covariance between models: 

![image](https://github.com/user-attachments/assets/91604c9c-7454-4022-a5fe-1c443aaab9dc)

#### Explanation of Terms:
- **$\text{Cov}(f_i(x), f_j(x))$**: The covariance between the predictions of models $f_i$ and $f_j$. If the models are highly correlated, the benefit of bagging is reduced because the errors made by different models are more likely to align.

- ### Random Forests for Climate Prediction and Uncertainty Quantification

In climate science, combining multiple models to improve the accuracy and reliability of predictions is a common approach. Random Forests, a popular machine learning algorithm, can play a key role in this process. In this blog, we'll proive a Python implementation that uses Random Forests to predict climate temperature anomalies, assess model performance, quantify uncertainties, and determine feature importance. 



```python
# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic data representing years
years = np.arange(1900, 2000)
n_years = len(years)

# Simulate outputs from three different climate models
# These are synthetic and for demonstration purposes
model1 = np.sin(0.02 * np.pi * years) + np.random.normal(0, 0.1, n_years)
model2 = np.cos(0.02 * np.pi * years) + np.random.normal(0, 0.1, n_years)
model3 = np.sin(0.02 * np.pi * years + np.pi/4) + np.random.normal(0, 0.1, n_years)

# Generate synthetic observed temperature anomalies
observed = (
    0.3 * model1 +
    0.5 * model2 +
    0.2 * model3 +
    np.random.normal(0, 0.05, n_years)
)

# Create a DataFrame to hold the data
data = pd.DataFrame({
    'Year': years,
    'Model1': model1,
    'Model2': model2,
    'Model3': model3,
    'Observed': observed
})

# Split the data into features (X) and target variable (y)
X = data[['Model1', 'Model2', 'Model3']]
y = data['Observed']

# Since we're dealing with time series data, use TimeSeriesSplit for cross-validation
tscv = TimeSeriesSplit(n_splits=5)

# Define parameter grid for hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# Initialize the Random Forest Regressor
rf = RandomForestRegressor(random_state=42)

# Initialize GridSearchCV with TimeSeriesSplit
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=tscv,
    scoring='neg_mean_squared_error',
    n_jobs=-1
)

# Fit GridSearchCV to find the best hyperparameters
grid_search.fit(X, y)

# Get the best estimator
best_rf = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)

# Use the last 20% of data as a test set to evaluate performance
split_index = int(n_years * 0.8)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

# Fit the best Random Forest model on the training data
best_rf.fit(X_train, y_train)

# Predict on the test set
y_pred = best_rf.predict(X_test)

# Calculate Mean Squared Error to evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.4f}")

# Quantify uncertainties by computing the standard deviation of predictions from all trees
all_tree_predictions = np.array([
    tree.predict(X_test.values) for tree in best_rf.estimators_
])

# Standard deviation across all trees for each prediction
prediction_std = np.std(all_tree_predictions, axis=0)

# Plot the observed vs. predicted values along with uncertainty bounds
plt.figure(figsize=(12, 6))
plt.plot(data['Year'], data['Observed'], label='Observed', color='black')
plt.plot(
    data['Year'][split_index:],
    y_pred,
    label='Predicted',
    color='blue'
)
plt.fill_between(
    data['Year'][split_index:],
    y_pred - prediction_std,
    y_pred + prediction_std,
    color='blue',
    alpha=0.2,
    label='Uncertainty'
)
plt.xlabel('Year')
plt.ylabel('Temperature Anomaly')
plt.title('Random Forest Prediction of Temperature Anomalies with Uncertainty')
plt.legend()
plt.grid(True)
plt.show()

# Feature Importance Analysis
importances = best_rf.feature_importances_
feature_names = X.columns
forest_importances = pd.Series(importances, index=feature_names)

# Plot feature importances
plt.figure(figsize=(8, 6))
forest_importances.sort_values().plot(kind='barh')
plt.title('Feature Importances')
plt.xlabel('Mean Decrease in Impurity')
plt.ylabel('Features')
plt.grid(True)
plt.show()
```

