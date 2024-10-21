# Understanding Forward and Backward Propagation: A Mathematical Perspective  
*By Kai Lin Woon*

## Introduction  
Artificial Neural Networks (ANNs) have revolutionized machine learning and artificial intelligence, enabling models to learn complex patterns from data. At the heart of training ANNs lies the **backpropagation algorithm**, a method used to compute gradients for updating the network's weights. 

This blog post delves into the mathematical underpinnings of **forward** and **backward propagation** with rigorous technical accuracy. We will then apply these concepts to a simple example: finding the gradient and intercept of a linear equation using gradient descent.

## 1. Forward Propagation

### Mathematical Formulation
Forward propagation is the process of computing the output of a neural network given an input by passing the data through each layer. For a network with $L$ layers, the forward propagation equations are as follows:

For layer $l = 1, 2, \dots, L$:

#### Linear Combination:
$$
z^{[l]} = W^{[l]} a^{[l-1]} + b^{[l]}
$$

- $W^{[l]}$: Weight matrix of shape $(n^{[l]}, n^{[l-1]})$.
- $b^{[l]}$: Bias vector of shape $(n^{[l]}, 1)$.
- $a^{[l-1]}$: Activation from the previous layer (or input $x$ when $l=1$).

#### Activation:
$$
a^{[l]} = \sigma^{[l]}(z^{[l]})
$$

- $\sigma^{[l]}$: Activation function for layer $l$ (e.g., sigmoid, ReLU).

#### Input Layer:
$$
a^{[0]} = x
$$

#### Output Layer:
The output of the network is $a^{[L]}$.

---

## 2. Backward Propagation

Backward propagation, or backpropagation, computes the gradients of the loss function with respect to each weight and bias in the network. These gradients are used to update the parameters during training.

### Derivation of the Backpropagation Algorithm
Consider a loss function $L(a^{[L]}, y)$, where $y$ is the true label.

#### Chain Rule of Calculus
The backpropagation algorithm relies heavily on the chain rule. For functions $f$ and $g$:

$$
\frac{\partial f(g(x))}{\partial x} = \frac{\partial f}{\partial g} \cdot \frac{\partial g}{\partial x}
$$

### Backpropagation Steps
For layer $l = L, L-1, \dots, 1$:

#### Compute Error Term ($\delta^{[l]}$):

For the output layer ($l=L$):

$$
\delta^{[L]} = \frac{\partial L}{\partial a^{[L]}} \odot \sigma'^{[L]}(z^{[L]})
$$

For hidden layers ($l < L$):

$$
\delta^{[l]} = (W^{[l+1]^T} \delta^{[l+1]}) \odot \sigma'^{[l]}(z^{[l]})
$$

Where:
- $\odot$: Element-wise multiplication.
- $\sigma'^{[l]}(z^{[l]})$: Derivative of the activation function.

### Compute Gradients:
- **Gradient w.r.t Weights**:
- 
$$
\frac{\partial L}{\partial W^{[l]}} = \delta^{[l]} a^{[l-1]^T}
$$

- **Gradient w.r.t Biases**:
- 
$$
\frac{\partial L}{\partial b^{[l]}} = \delta^{[l]}
$$

### Computing Gradients
The goal is to compute $\frac{\partial L}{\partial W^{[l]}}$ and $\frac{\partial L}{\partial b^{[l]}}$ for all layers $l$.

#### Initialize:
Start from the output layer and compute $\delta^{[L]}$.

#### Recursive Computation:
For each layer $l$, compute $\delta^{[l]}$ using $\delta^{[l+1]}$ and $W^{[l+1]}$.

#### Gradient Calculation:
Use $\delta^{[l]}$ to compute gradients w.r.t $W^{[l]}$ and $b^{[l]}$.

## 3. Application: Linear Regression

### Model Representation
Linear regression aims to model the relationship between a scalar response $y$ and a scalar predictor $x$ using a linear function:

$$
y = mx + c
$$

Where:
- $m$: Slope (gradient).
- $c$: Intercept.

We can represent linear regression as a neural network without hidden layers:

- **Input**: $x$
- **Output**: $\hat{y} = a$
- **Parameters**: $W = m$, $b = c$

---

### Forward and Backward Propagation in Linear Regression

#### Forward Propagation
**Compute Predicted Output**:

$$
\hat{y} = Wx + b
$$

**Compute Loss** (Using Mean Squared Error (MSE)):

$$
L(\hat{y}, y) = \frac{1}{2} (\hat{y} - y)^2
$$

---

#### Backward Propagation
Compute gradients of the loss function w.r.t $W$ and $b$.

**Gradient w.r.t Output**:

$$
\frac{\partial L}{\partial \hat{y}} = \hat{y} - y
$$

**Gradients w.r.t Parameters**:

- **Weight Gradient**:
  
$$
\frac{\partial L}{\partial W} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial W} = (\hat{y} - y)x
$$

- **Bias Gradient**:

$$
\frac{\partial L}{\partial b} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial b} = (\hat{y} - y)
$$

---

### Example Calculation
Let's walk through an example with a single data point.

Given:
- **Data point**: $x = 2$, $y = 5$
- **Initial parameters**: $W = 0.5$, $b = 0.0$
- **Learning rate**: $\alpha = 0.1$

#### Step 1: Forward Propagation
**Compute predicted output**:

$$
\hat{y} = Wx + b = 0.5 \times 2 + 0 = 1.0
$$

**Compute loss**:

$$
L = \frac{1}{2} (\hat{y} - y)^2 = \frac{1}{2} (1.0 - 5)^2 = \frac{1}{2} (16) = 8.0
$$

#### Step 2: Backward Propagation
**Compute gradients**:

- **Gradient w.r.t Output**:
  
$$
\frac{\partial L}{\partial \hat{y}} = \hat{y} - y = 1.0 - 5 = -4.0
$$

- **Gradient w.r.t Weight**:
  
$$
\frac{\partial L}{\partial W} = (\hat{y} - y)x = -4.0 \times 2 = -8.0
$$

- **Gradient w.r.t Bias**:
  
$$
\frac{\partial L}{\partial b} = \hat{y} - y = -4.0
$$

#### Step 3: Parameter Update
Update parameters using gradient descent:

- **Update Weight**:

$$
W_{\text{new}} = W_{\text{old}} - \alpha \frac{\partial L}{\partial W} = 0.5 - 0.1 \times (-8.0) = 0.5 + 0.8 = 1.3
$$

- **Update Bias**:

$$
b_{\text{new}} = b_{\text{old}} - \alpha \frac{\partial L}{\partial b} = 0.0 - 0.1 \times (-4.0) = 0.0 + 0.4 = 0.4
$$

#### Step 4: Repeat Iterations
Perform additional iterations to further minimize the loss.

### 4. Python Implementation: Finding the Gradient and Intercept using Backpropagation
Now, let's implement the forward and backpropagation algorithm in Python to fit a linear regression model by finding the optimal values for $m$ (slope) and $b$ (intercept).

```python
import numpy as np

# Mean Squared Error loss function
def compute_loss(y, y_hat):
    m = y.shape[0]
    loss = (1 / (2 * m)) * np.sum((y_hat - y) ** 2)
    return loss

# Forward propagation to compute predictions
def forward_propagation(X, m, b):
    # Calculate the predicted output (y_hat = mx + b)
    y_hat = m * X + b
    return y_hat

# Backward propagation to compute gradients
def backward_propagation(X, y, y_hat):
    m = X.shape[0]
    
    # Gradient of loss w.r.t. m (slope)
    d_m = (-2 / m) * np.sum(X * (y - y_hat))
    
    # Gradient of loss w.r.t. b (intercept)
    d_b = (-2 / m) * np.sum(y - y_hat)
    
    return d_m, d_b

# Gradient Descent algorithm to update m and b
def gradient_descent(X, y, m, b, learning_rate, num_iterations):
    for i in range(num_iterations):
        # Forward propagation to calculate predictions
        y_hat = forward_propagation(X, m, b)
        
        # Compute loss
        loss = compute_loss(y, y_hat)
        
        # Backward propagation to compute gradients
        d_m, d_b = backward_propagation(X, y, y_hat)
        
        # Update m and b using the gradients
        m -= learning_rate * d_m
        b -= learning_rate * d_b
        
        # Print the loss every 100 iterations
        if i % 100 == 0:
            print(f"Iteration {i}, Loss: {loss}, Slope (m): {m}, Intercept (b): {b}")
    
    return m, b

# Example usage
if __name__ == "__main__":
    # Generate some synthetic data for linear regression
    np.random.seed(0)
    X = 2 * np.random.rand(100, 1)  # 100 data points (features)
    y = 4 + 3 * X + np.random.randn(100, 1)  # y = 4 + 3x + noise
    
    # Initial values for m and b
    m_init = 0
    b_init = 0
    
    # Hyperparameters
    learning_rate = 0.01
    num_iterations = 1000
    
    # Perform gradient descent to find the optimal m and b
    m_optimal, b_optimal = gradient_descent(X, y, m_init, b_init, learning_rate, num_iterations)
    
    print(f"\nOptimal Slope (m): {m_optimal}, Optimal Intercept (b): {b_optimal}")
```
### Explanation of Code:

#### Initialization:
- We start with randomly generated data $X$ and $y$, where $y$ is linearly dependent on $X$ with some added noise.
- The slope $m$ and intercept $b$ are initialized to 0.

#### Forward Propagation:
- In each iteration of gradient descent, we compute the predicted output $\hat{y}$ using the current values of $m$ and $b$. This is a straightforward calculation of $\hat{y} = mx + b$.

#### Loss Calculation:
- The loss function used is the Mean Squared Error (MSE), which is calculated as the average of the squared differences between the predicted and actual values.

#### Backward Propagation:
- The gradients $\frac{\partial J}{\partial m}$ and $\frac{\partial J}{\partial b}$ are computed using the formulas derived from the chain rule. These gradients tell us how to adjust $m$ and $b$ to minimize the loss.

#### Gradient Descent:
- We update the values of $m$ and $b$ using the gradients. The learning rate $\alpha$ controls how much the parameters are adjusted in each iteration. The process is repeated for a specified number of iterations.

#### Result:
- After the gradient descent process is completed, the final values of $m$ and $b$ will be close to the optimal values that minimize the loss.

