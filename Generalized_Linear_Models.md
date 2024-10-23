# **Generalized Linear Models in Machine Learning: A Technical Guide**
*By Kai Lin Woon*
## 1. Introduction to Generalized Linear Models

Generalized Linear Models (GLMs) are a class of flexible models that extend ordinary linear regression to accommodate various types of response variables, including those that are not normally distributed. GLMs generalize linear regression by allowing for:

- **A linear predictor:** This is a linear combination of the input variables.
- **A link function:** This function maps the linear predictor to the expected value of the response variable.
- **A distribution from the exponential family:** This includes distributions such as normal, binomial, Poisson, and gamma distributions.

## 2. Components of a Generalized Linear Model

A GLM consists of three primary components:

### **Random Component:**

Specifies the probability distribution of the response variable \$Y\$. In GLMs, this distribution must belong to the exponential family of distributions.

### **Systematic Component:**

Describes the linear predictor, which is the linear combination of the explanatory variables:

$$
\eta = \beta_0 + \beta_1 x_1 + \cdots + \beta_p x_p
$$

Here, \$\eta\$ is the linear predictor, and \$x_1, x_2, \dots, x_p\$ are the input features.

### **Link Function:**

A function \$g(\cdot)\$ that relates the expected value of the response variable \$E(Y) = \mu\$ to the linear predictor \$\eta\$.

$$
g(\mu) = \eta
$$

Different link functions allow us to model various types of data. For example, a **logit link function** is used for binary classification (logistic regression).

# **Section 3: The Exponential Family of Distributions**

A key feature of Generalized Linear Models is that they can handle response variables that follow any distribution from the exponential family of distributions. This family includes many commonly used distributions, such as the normal, binomial, Poisson, and gamma distributions. The form of any exponential family distribution is given by:

$$
f(y; \theta, \phi) = \exp\left( \frac{y\theta - b(\theta)}{\phi} + c(y, \phi) \right)
$$

This general form defines a family of distributions that can be used in GLMs, and it includes:

- \$\theta\$: the natural parameter or canonical parameter of the distribution, which relates to the mean of the distribution.
- \$\phi\$: the dispersion parameter, which controls the variance (in some distributions, this may be constant or ignored).
- \$b(\theta)\$: a function that depends on \$\theta\$, which helps define the mean of the distribution.
- \$c(y, \phi)\$: a function that depends on the observed response \$y\$ and the dispersion parameter \$\phi\$. This term ensures that the probability function sums or integrates to 1.

Let’s break down how this applies to some specific distributions in the exponential family.

### **Normal (Gaussian) Distribution**

The normal distribution is used for continuous data, typically in standard linear regression. The probability density function of the normal distribution is:

$$
f(y; \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left( -\frac{(y - \mu)^2}{2\sigma^2} \right)
$$

Here, the natural parameter \$\theta = \mu\$ (the mean), and the dispersion parameter \$\phi = \sigma^2\$ (the variance). The normal distribution has the form of the exponential family with:

- \$\theta = \mu\$
- \$b(\theta) = \frac{\theta^2}{2}\$
- \$c(y, \phi) = -\frac{y^2}{2\phi} - \frac{1}{2} \log(2\pi\phi)\$

### **Binomial Distribution**

The binomial distribution is used for binary or categorical data, as in logistic regression. The probability mass function for the binomial distribution is:

$$
f(y; p, n) = \binom{n}{y} p^y (1 - p)^{n - y}
$$

For a single trial (\$n = 1\$), this simplifies to:

$$
f(y; p) = p^y (1 - p)^{1 - y}
$$

Where \$y\$ is either 0 or 1. In this case:

- \$\theta = \log\left(\frac{p}{1 - p}\right)\$ (the log odds, or logit),
- \$b(\theta) = \log(1 + e^\theta)\$,
- \$c(y) = 0\$.

### **Poisson Distribution**

The Poisson distribution is used for modeling count data. The probability mass function for the Poisson distribution is:

$$
f(y; \lambda) = \frac{\lambda^y e^{-\lambda}}{y!}
$$

Where \$\lambda\$ is the expected number of events. In this case:

- \$\theta = \log(\lambda)\$ (log of the mean),
- \$b(\theta) = e^\theta\$,
- \$c(y) = -\log(y!)\$.

### **Why Exponential Family?**

GLMs can handle these different distributions because the exponential family allows us to maintain mathematical consistency when modeling the relationship between the predictors and the response variable. This uniform structure helps us generalize the process of estimation (through maximum likelihood) for different types of response variables.

# **Section 4: Link Functions**

The link function in a GLM is a crucial component that maps the expected value of the response variable to the linear predictor. Recall that the linear predictor is:

$$
\eta = X\beta = \beta_0 + \beta_1 x_1 + \cdots + \beta_p x_p
$$

However, for many distributions (e.g., binomial or Poisson), the mean response (\$\mu\$) is not directly related to \$\eta\$, so we need a transformation to ensure that the predictions are valid. This transformation is the link function \$g(\mu)\$.

The choice of link function depends on the distribution of the response variable, and different link functions are appropriate for different distributions. The inverse of the link function, \$g^{-1}(\eta)\$, gives us the predicted mean \$\mu\$ from the linear predictor.

### **Common Link Functions:**

- **Identity Link:** \$g(\mu) = \mu\$

  This is used for the normal (Gaussian) distribution, which is typical in linear regression. The identity link means that the expected value of the response variable is directly modeled as a linear combination of the predictors:

$$
\mu = \eta = X\beta
$$

- **Logit Link:** \$g(\mu) = \log\left(\frac{\mu}{1 - \mu}\right)\$

  This is used for the binomial distribution, especially in logistic regression. The logit link ensures that the predicted probabilities (\$\mu\$) are constrained to be between 0 and 1:

$$
\mu = \frac{1}{1 + e^{-\eta}}
$$

  This is the inverse logit function, which is also known as the sigmoid function. It maps any real number to the range \$(0, 1)\$, making it ideal for binary classification problems.

- **Log Link:** \$g(\mu) = \log(\mu)\$

  This is used for the Poisson distribution and is common in Poisson regression. The log link ensures that the predicted mean \$\mu\$ is positive, which is important for count data:

$$
\mu = e^{\eta}
$$

### **Derivation of the Link Function and Linear Predictor Relationship**

In GLMs, the expected value of the response variable \$\mu\$ is related to the linear predictor \$\eta\$ through the link function:

$$
g(\mu) = \eta
$$

Therefore, the expected value of the response variable is:

$$
\mu = g^{-1}(\eta)
$$

For example, in logistic regression, we use the logit link:

$$
\log\left(\frac{\mu}{1 - \mu}\right) = \eta
$$

Solving for \$\mu\$, we get the logistic (sigmoid) function:

$$
\mu = \frac{1}{1 + e^{-\eta}}
$$

In this case, \$\mu\$ represents the probability that \$Y = 1\$.

Similarly, for Poisson regression, we use the log link:

$$
\log(\mu) = \eta
$$

Solving for \$\mu\$, we get:

$$
\mu = e^{\eta}
$$

This ensures that the predicted mean of the count variable is always positive.

# **5. Mathematical Derivation of GLMs**

Now let’s delve into the derivation of GLMs, beginning with the log-likelihood of the exponential family of distributions. Given the form of the exponential family:

$$
\log f(y; \theta, \phi) = \frac{y\theta - b(\theta)}{\phi} + c(y, \phi)
$$

The goal is to estimate the parameters \$\beta_0, \beta_1, \dots, \beta_p\$, which are related to the linear predictor \$\eta = X\beta\$ through the link function \$g(\mu)\$.

For estimation, we usually maximize the log-likelihood function. For \$n\$ observations, the log-likelihood is given by:

$$
\ell(\beta) = \sum_{i=1}^{n} \log f(y_i; \theta_i, \phi)
$$

Where \$\theta_i = g^{-1}(\eta_i)\$.

The score function (the derivative of the log-likelihood with respect to \$\beta\$) is:

$$
U(\beta) = \frac{\partial \ell(\beta)}{\partial \beta} = X^T (Y - \mu)
$$

Where \$X\$ is the matrix of input features, and \$Y\$ is the vector of observed responses.

The Fisher information matrix is:

$$
I(\beta) = -\mathbb{E} \left[ \frac{\partial^2 \ell(\beta)}{\partial \beta^2} \right] = X^T W X
$$

Where \$W\$ is a diagonal matrix of weights determined by the variance function of the exponential family.

The solution to the maximum likelihood estimation is typically obtained via **Iteratively Reweighted Least Squares (IRLS)**, which iterates between updating the linear predictor \$\eta\$ and the parameters \$\beta\$.

# **6. Example: Logistic Regression (A Special Case of GLM)**

Logistic Regression is a type of GLM where the response variable is binary (0 or 1), and the link function is the **logit function**:

$$
g(\mu) = \log \left( \frac{\mu}{1 - \mu} \right)
$$

This gives the relationship:

$$
\eta = X\beta = \log \left( \frac{\mu}{1 - \mu} \right)
$$

Solving for \$\mu\$, we get:

$$
\mu = \frac{1}{1 + e^{-\eta}}
$$

Where \$\mu\$ is the probability that the response variable \$Y = 1\$, given the predictors.

# **7. Example: Logistic Regression (A Special Case of GLM)**
If we want to implement logistic regression from scratch without using `statsmodels.api` or any machine learning library (except for loading the raw Iris dataset), we will need to manually implement:

1. The **sigmoid function**.
2. **Gradient descent** to optimize the logistic regression cost function.
3. **Binary classification** using logistic regression.

Here is the manual implementation of logistic regression:

```python
# Import necessary libraries
import numpy as np
from sklearn.datasets import load_iris

# Load the iris dataset
data = load_iris()
X = data.data[data.target != 2]  # Keep only the first two classes (Setosa and Versicolor)
y = data.target[data.target != 2]

# Manually split the data into training and testing sets (70% train, 30% test)
np.random.seed(42)  # For reproducibility
indices = np.random.permutation(len(X))  # Shuffle indices
train_size = int(0.7 * len(X))  # Define the size for the training set

# Split the data
X_train, X_test = X[indices[:train_size]], X[indices[train_size:]]
y_train, y_test = y[indices[:train_size]], y[indices[train_size:]]

# Normalize the features (optional but recommended for gradient descent)
X_train = (X_train - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)
X_test = (X_test - np.mean(X_test, axis=0)) / np.std(X_test, axis=0)

# Add a bias term (intercept) to X_train and X_test (a column of ones)
X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Logistic regression cost function
def compute_cost(X, y, theta):
    m = len(y)
    h = sigmoid(X @ theta)
    epsilon = 1e-5  # Small constant to avoid log(0)
    cost = -(1 / m) * (y.T @ np.log(h + epsilon) + (1 - y).T @ np.log(1 - h + epsilon))
    return cost

# Gradient descent to optimize theta
def gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)
    cost_history = np.zeros(iterations)
    
    for i in range(iterations):
        # Calculate the hypothesis
        h = sigmoid(X @ theta)
        
        # Update the parameters (gradient descent step)
        theta -= (alpha / m) * (X.T @ (h - y))
        
        # Save the cost for this iteration
        cost_history[i] = compute_cost(X, y, theta)
    
    return theta, cost_history

# Initialize the parameters (theta)
theta = np.zeros(X_train.shape[1])

# Set hyperparameters for gradient descent
alpha = 0.1  # Learning rate
iterations = 1000  # Number of iterations for gradient descent

# Run gradient descent
theta, cost_history = gradient_descent(X_train, y_train, theta, alpha, iterations)

# Print final cost after training
print(f"Final cost: {cost_history[-1]}")

# Make predictions on the test set
y_pred_prob = sigmoid(X_test @ theta)
y_pred = (y_pred_prob >= 0.5).astype(int)

# Calculate accuracy on the test set
accuracy = np.mean(y_pred == y_test)
print(f"Test accuracy: {accuracy:.4f}")
```

## Explanation of the Iris Dataset

The Iris dataset is one of the most well-known datasets in machine learning and statistics. It is commonly used for classification problems and contains data about three species of Iris flowers: **Iris setosa**, **Iris versicolor**, and **Iris virginica**. The dataset has 150 samples and 4 features (measurements) for each flower:

- **Sepal length (cm)**
- **Sepal width (cm)**
- **Petal length (cm)**
- **Petal width (cm)**

Additionally, each sample is labeled with the corresponding species of the flower, which is the target variable for classification.

### The target variable has three classes (species):

- `0`: Iris Setosa
- `1`: Iris Versicolor
- `2`: Iris Virginica

We filter the dataset to only include the first two classes (**Setosa** and **Versicolor**), making the problem a **binary classification task**. This is because logistic regression (a GLM with a binomial distribution) is typically used for binary classification, where the response variable is either `0` or `1`.
## Explanation of the Code:

### **Data Loading and Preprocessing:**

- We load the Iris dataset using `sklearn.datasets.load_iris()`.
- We filter the dataset to include only the first two classes (Setosa and Versicolor) for binary classification.
- We manually split the data into a training set (70%) and a test set (30%) by shuffling the indices and slicing the dataset.

### **Normalization:**

- Normalization (optional but recommended) is applied to scale the features of both the training and test sets to have zero mean and unit variance. This is especially important for gradient-based optimization methods like gradient descent.

### **Bias Term (Intercept):**

- We add a bias term (also called an intercept) to the feature matrices by inserting a column of ones at the beginning of `X_train` and `X_test`. This is needed for the logistic regression model because we want an intercept in our model.

### **Sigmoid Function:**

The sigmoid function is used to map the linear combination of input features (weighted by coefficients) to a probability:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

Where:

$$
z = X\theta
$$

is the linear combination of the features and their corresponding weights (parameters \$\theta\$).

### **Cost Function:**

The cost function for logistic regression is the logistic loss function (also known as cross-entropy loss):

$$
J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y_i \log(h_{\theta}(x_i)) + (1 - y_i) \log(1 - h_{\theta}(x_i)) \right]
$$

Where:
- \$h_{\theta}(x_i)\$ is the predicted probability for sample \$i\$,
- \$y_i\$ is the actual label (0 or 1).

### **Gradient Descent:**

Gradient descent is used to minimize the cost function and find the optimal parameters (\$\theta\$). In each iteration, the parameters are updated based on the gradient of the cost function with respect to \$\theta\$:

$$
\theta := \theta - \alpha \frac{1}{m} \sum_{i=1}^{m} \left( h_{\theta}(x_i) - y_i \right) x_i
$$

Where:
- \$\alpha\$ is the learning rate (controls the step size in each iteration),
- \$m\$ is the number of training examples,
- \$h_{\theta}(x_i)\$ is the predicted probability for sample \$i\$,
- \$x_i\$ is the feature vector for sample \$i\$.

### **Making Predictions:**

After training the model, we use the learned parameters \$\theta\$ to make predictions on the test set. The predicted probability is computed using the sigmoid function:

$$
y_{\text{pred\text{-}prob}} = \sigma(X_{\text{test}} \theta)
$$


To convert these probabilities into class predictions (0 or 1), we use a threshold of 0.5. If:

$$
y_{\text{pred\text{-}prob}} \geq 0.5
$$

predict 1, otherwise predict 0.

### **Evaluating the Model:**

The accuracy is computed as the fraction of correct predictions on the test set:

$$
\text{Accuracy} = \frac{\text{Number of correct predictions}}{\text{Total number of predictions}}
$$

### **Key Steps in the Implementation:**

1. **Sigmoid Function:** A simple non-linear activation function that maps any real number to a value between 0 and 1, making it suitable for binary classification.
2. **Cost Function:** The cross-entropy loss measures how well the logistic regression model fits the data.
3. **Gradient Descent:** Iteratively minimizes the cost function by adjusting the model parameters to reduce the prediction error.
4. **Manual Predictions:** After training, we compute the predictions by applying the learned model parameters to the test data.

### **Final Accuracy:**

The final test accuracy will vary based on how well the model converges during gradient descent and how well it generalizes to the test data. In this case, since the data is highly separable (especially for the Setosa and Versicolor classes in the Iris dataset), we expect the accuracy to be relatively high.

This code manually implements logistic regression from scratch without using any machine learning libraries for the model itself, only for data loading.

## Example: Regularized Logistic Regression with sklearn libaries for fast implementation

We can use logistic regression with regularization from `scikit-learn`, which is easier to handle than using `statsmodels` for GLM when regularization is needed. Here is an example:

```python
# Import required libraries
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load the iris dataset
data = load_iris()
X = data.data[data.target != 2]  # Keep only the first two classes (Setosa and Versicolor)
y = data.target[data.target != 2]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Fit logistic regression with L2 regularization
model = LogisticRegression(penalty='l2', solver='lbfgs')
model.fit(X_train, y_train)

# Print the accuracy on the test set
print(f"Test accuracy: {model.score(X_test, y_test):.4f}")
```
Let's break down the Regularized Logistic Regression example in detail and explain each part of the code, step by step.

## 1. Importing Libraries

In this section, we import the necessary libraries for building a logistic regression model.

- **LogisticRegression:** This is the logistic regression model from `scikit-learn`. We will use this to perform regularized logistic regression with **L2 regularization**.
- **load_iris:** This function from `sklearn.datasets` loads the **Iris dataset**, which we will use for binary classification.
- **train_test_split:** This function helps split the dataset into **training** and **testing sets**, which is a common practice in machine learning to evaluate model performance.

## 2. Loading and Preprocessing the Iris Dataset

Here, we load the Iris dataset and preprocess it to suit the binary classification task.

- `data = load_iris()`: This loads the entire Iris dataset into the variable `data`. It includes:
  - Features: `data.data` (sepal length, sepal width, petal length, and petal width).
  - Target labels: `data.target` (species labels).
  
- `X = data.data[data.target != 2]`: This filters the dataset to include only the first two classes, **Setosa** and **Versicolor**, by excluding the third class **Iris Virginica** (which is labeled as 2). Logistic regression in this example is used for binary classification, so we only need two classes.
  - `data.data` contains the input features.
  - `data.target` contains the labels corresponding to the flower species: `0` for Setosa, `1` for Versicolor, and `2` for Virginica.
  - We use `data.target != 2` to create a mask that filters out rows where the target class is `2` (Virginica).

- `y = data.target[data.target != 2]`: This extracts the target labels (`0` for Setosa and `1` for Versicolor) for the remaining samples after filtering.

Now, the feature matrix `X` contains only samples for two classes (**Setosa** and **Versicolor**), and the target vector `y` contains the corresponding class labels (`0` or `1`).
## 3. Splitting the Dataset

Here, we split the filtered dataset into training and testing sets using `train_test_split()`.

- **`X_train, X_test`**: These are the training and testing sets of the feature matrix \$ X \$. The training set is used to train the logistic regression model, while the test set is used to evaluate its performance.
  
- **`y_train, y_test`**: These are the corresponding labels for the training and testing sets.
  
- **`test_size=0.3`**: This argument specifies that 30% of the dataset will be used for testing, and the remaining 70% will be used for training.
  
- **`random_state=42`**: This ensures that the split is reproducible. By setting a random seed (42 in this case), we ensure that every time the code is run, the split will be the same, making the results consistent.
- 
## 4. Fitting Logistic Regression Model with Regularization

In this part, we create and fit a regularized logistic regression model.

- **`LogisticRegression(penalty='l2', solver='lbfgs')`:**
  - `penalty='l2'`: This specifies that **L2 regularization** (Ridge regularization) is applied to the logistic regression model. L2 regularization helps to avoid overfitting and perfect separation by shrinking the coefficients and preventing them from becoming too large.
  
  - **L2 regularization** adds a term to the loss function, penalizing large coefficient values, and helps in preventing overfitting. The regularized objective function is:

$$
L(\beta) = - \sum_{i=1}^{n} \left[ y_i \log(p_i) + (1 - y_i) \log(1 - p_i) \right] + \lambda \sum_{j=1}^{p} \beta_j^2
$$

  Where:
  - \$\lambda\$ is the regularization strength, and
  - \$\beta_j\$ are the coefficients.
  
  The second term in this expression is the regularization term, which penalizes large values of the coefficients.

- **`solver='lbfgs'`:** This is the optimization algorithm used to solve the logistic regression problem. **lbfgs** is a quasi-Newton method that is well-suited for small- to medium-sized datasets. It is efficient and handles L2 regularization well.

- **`model.fit(X_train, y_train)`**: This method fits the logistic regression model to the training data (`X_train` and `y_train`). The model learns the parameters (coefficients \$\beta\$) by maximizing the log-likelihood function with L2 regularization.

## 5. Evaluating the Model

Finally, we evaluate the model’s performance on the test data.

- **`model.score(X_test, y_test)`**: This method calculates the accuracy of the fitted model on the test data. Accuracy is defined as the proportion of correctly classified samples:

$$
\text{Accuracy} = \frac{\text{Number of correct predictions}}{\text{Total number of predictions}}
$$

For a binary classification problem like this, it checks whether the predicted label matches the true label in the test set and calculates the percentage of correct predictions.

The result is printed in the format `Test accuracy: X.XXXX`, where the accuracy is shown to four decimal places.

### Detailed Explanation of Key Concepts

#### **L2 Regularization**
- **L2 regularization** adds a penalty term proportional to the square of the magnitude of the coefficients. This ensures that the model does not rely too heavily on any particular feature and prevents coefficients from becoming excessively large, which is important when perfect separation occurs in the data.
  
- In logistic regression, without regularization, if the data can be perfectly separated (i.e., there exists a linear boundary that classifies all samples correctly), the coefficients can go to infinity. **L2 regularization** mitigates this issue by controlling the magnitude of the coefficients, keeping the model more generalizable.

#### **Train-Test Split**
- The train-test split is important for evaluating the model's generalization ability. By training the model on one portion of the data (70% in this case) and testing it on a separate portion (30%), we can see how well the model performs on unseen data.
  
- If the model performs well on both the training and test sets, it is likely to generalize well to new, unseen data.

#### **Model Evaluation**
- **Accuracy** is a straightforward metric for evaluating the model in binary classification tasks. However, accuracy alone may not always be sufficient. For example, in cases of imbalanced datasets, other metrics such as **precision**, **recall**, or the **F1 score** might provide better insight into model performance.

### Summary of Code Flow

1. The Iris dataset is loaded, and we filter it to retain only two classes (Setosa and Versicolor), which is appropriate for binary classification.
2. The dataset is split into training and testing sets using a 70-30 split.
3. A logistic regression model with **L2 regularization** (Ridge) is created and trained using the training data.
4. The trained model is evaluated using the test data, and the accuracy is printed.
