# Understanding the Levenberg-Marquardt Algorithm

## Introduction

The **Levenberg-Marquardt (LM) Algorithm** is a widely used optimization method, primarily designed for solving non-linear least squares problems. It is particularly useful when you need to fit a mathematical model to a set of observations. The LM algorithm is a combination of two different techniques: the **Gauss-Newton algorithm** and **gradient descent**.

In this blog post, we will explore how the Levenberg-Marquardt algorithm works, why it’s useful, and where it is applied in real-world scenarios.

---

## Section 1: The Problem – Non-linear Least Squares

At its core, the LM algorithm solves the following problem: given a set of measurements $(y_i)$, and a model function $(f(x, \beta))$ where:

- $(x)$ is the input (independent variable),
- $(\beta)$ is the parameter vector,
- $(y)$ is the output (dependent variable),

We want to find the parameter vector $(\beta)$ that minimizes the sum of squared differences between the observed values $(y_i)$ and the model's predicted values $(f(x_i, \beta))$:

$$
S(\beta) = \sum_{i=1}^{n} [y_i - f(x_i, \beta)]^2
$$

The solution to this problem is the set of parameters $(\beta)$ that minimizes the sum of squared residuals.

---

## Section 2: Combining Gauss-Newton and Gradient Descent

The LM algorithm is a hybrid of two optimization methods:

- **Gauss-Newton Algorithm**: Efficient for solving non-linear least squares when the residuals are small.
- **Gradient Descent**: Useful for steep descent, especially in the early iterations when far from the solution.

The key idea of LM is to switch between these methods based on the behavior of the residuals.

### Gauss-Newton Update Rule

The Gauss-Newton method approximates the solution by linearizing the function around the current estimate of $(\beta)$ and solving iteratively. It updates the parameters as:

$$
\beta_{new} = \beta_{old} - (J^T J)^{-1} J^T r
$$

Where:

- $(J)$ is the Jacobian matrix of partial derivatives of the residuals with respect to $(\beta)$,
- $(r)$ is the vector of residuals $(y_i - f(x_i, \beta))$.

This works well when close to the minimum but can fail when far from it, leading to non-converging steps.

### Gradient Descent Update Rule

The gradient descent method updates the parameters in the opposite direction of the gradient of the cost function:

$$
\beta_{new} = \beta_{old} - \alpha \nabla S(\beta)
$$

Where:

- $(\alpha)$ is the step size (learning rate),
- $(\nabla S(\beta))$ is the gradient of the cost function with respect to $(\beta)$.

This method ensures the algorithm converges but can be slow, especially near the minimum.

---

## Section 3: The Levenberg-Marquardt Algorithm

The LM algorithm balances between Gauss-Newton and gradient descent by introducing a damping parameter $(\lambda)$. The parameter controls whether the algorithm behaves more like Gauss-Newton or gradient descent at each iteration.

The update rule for LM is:

$$
\beta_{new} = \beta_{old} - (J^T J + \lambda I)^{-1} J^T r
$$

Where:

- $(I)$ is the identity matrix,
- $(\lambda)$ is the damping factor that is adjusted at each iteration.

### Adjusting the Damping Parameter

- If $(\lambda)$ is large, the algorithm behaves more like gradient descent, taking small steps in the direction of the gradient.
- If $(\lambda)$ is small, it behaves more like Gauss-Newton, using the curvature of the error surface to take larger steps.

### Iterative Process

1. **Initialize** the parameters $(\beta)$ and set $(\lambda)$ to a large value.
2. **Compute** the Jacobian matrix $(J)$ and residual vector $(r)$.
3. **Update** $(\beta)$ using the LM update rule.
4. **Evaluate** the error $(S(\beta))$.
   - If the error is reduced, **decrease** $(\lambda)$ (move towards Gauss-Newton behavior).
   - If the error increases, **increase** $(\lambda)$ (move towards gradient descent).
5. **Repeat** until convergence.

---

## Section 4: Advantages of the Levenberg-Marquardt Algorithm

1. **Combines the Best of Both Worlds**: By blending Gauss-Newton and gradient descent, LM is robust for a wide range of problems, especially when far from the solution.
2. **Efficient for Non-Linear Least Squares**: The algorithm is optimized for problems where the cost function is the sum of squared residuals.
3. **Adaptable**: The adjustment of the damping factor makes the algorithm flexible and capable of navigating complex error surfaces.

---

## Section 5: Applications

The LM algorithm is commonly used in various fields including:

- **Curve fitting**: When fitting models to data points, such as in regression analysis.
- **Machine learning**: For optimizing parameters of models such as neural networks, particularly when training involves non-linear error surfaces.
- **Computer vision**: Used in tasks like bundle adjustment in structure-from-motion problems.
- **Control systems**: For system identification and parameter estimation.

---

## Section 6: Python Example of the LM Algorithm

We aim to fit an exponential model of the form:

$$
f(x, \beta) = \beta_0 e^{\beta_1 x}
$$

Where:

- $( x )$ is the independent variable,
- $( \beta = [\beta_0, \beta_1] )$ are the parameters we need to estimate,
- $( f(x, \beta) )$ is the model we want to fit to the data.

Given a set of observed data points $( (x_i, y_i) )$, we want to minimize the sum of squared residuals:

$$
S(\beta) = \sum_{i=1}^{n} \left( y_i - f(x_i, \beta) \right)^2
$$

---

## Section 2: The Levenberg-Marquardt Algorithm

The LM algorithm iteratively adjusts the parameters $(\beta)$ to minimize the sum of squared residuals. It blends the Gauss-Newton method and gradient descent by introducing a **damping factor** $(\lambda)$, which controls the step size.

The update rule for the parameters $(\beta)$ is:

$$
\beta_{new} = \beta_{old} - (J^T J + \lambda I)^{-1} J^T r
$$

Where:

- $( J )$ is the Jacobian matrix of partial derivatives of the model with respect to $(\beta)$,
- $( r )$ is the residual vector (difference between observed and predicted values),
- $( \lambda )$ is the damping factor.

## Section 3: Python Implementation of the Levenberg-Marquardt Algorithm

Below is the Python code for implementing the LM algorithm without using `scipy`. We manually compute the Jacobian matrix and iteratively adjust the damping factor to converge on the optimal parameters.

```python
import numpy as np

def model(x, beta):
    """Model function: Exponential function."""
    return beta[0] * np.exp(beta[1] * x)

def residuals(beta, x, y):
    """Calculate residuals: Difference between observed and predicted."""
    return y - model(x, beta)

def jacobian(x, beta):
    """Jacobian matrix of partial derivatives of the model w.r.t beta."""
    J = np.zeros((x.size, len(beta)))
    J[:, 0] = np.exp(beta[1] * x)        # Derivative w.r.t beta[0]
    J[:, 1] = beta[0] * x * np.exp(beta[1] * x)  # Derivative w.r.t beta[1]
    return J

def levenberg_marquardt(x, y, beta_init, max_iters=100, tol=1e-6, lambda_init=0.01):
    """Levenberg-Marquardt optimization algorithm."""
    beta = np.array(beta_init)
    lambda_factor = 10
    lambda_param = lambda_init
    prev_error = np.inf

    for iteration in range(max_iters):
        # Compute residuals and Jacobian matrix
        r = residuals(beta, x, y)
        J = jacobian(x, beta)

        # Compute the normal equation components
        JTJ = np.dot(J.T, J)             # J^T J
        JTr = np.dot(J.T, r)             # J^T r

        # Augment with the damping factor
        H = JTJ + lambda_param * np.eye(len(beta))

        # Solve for the parameter update (delta)
        delta = np.linalg.solve(H, JTr)

        # Update beta parameters
        beta_new = beta - delta

        # Calculate the new residuals
        r_new = residuals(beta_new, x, y)
        error_new = np.sum(r_new**2)

        # Check for improvement
        if error_new < prev_error:
            # If improvement, accept the new parameters and decrease lambda
            beta = beta_new
            lambda_param /= lambda_factor
            prev_error = error_new
            print(f"Iteration {iteration+1}: Error = {error_new}")
        else:
            # If no improvement, increase lambda (move towards gradient descent)
            lambda_param *= lambda_factor

        # Check convergence
        if np.linalg.norm(delta) < tol:
            print(f"Converged in {iteration+1} iterations")
            break

    return beta

# Example data for fitting
x_data = np.linspace(0, 10, 100)
true_beta = [3, 0.5]
y_data = model(x_data, true_beta) + np.random.normal(0, 0.5, size=x_data.size)

# Initial guess for parameters
initial_beta = [1, 0.1]

# Run the Levenberg-Marquardt algorithm
fitted_beta = levenberg_marquardt(x_data, y_data, initial_beta)

print("Fitted parameters:", fitted_beta)
```
---
## Conclusion 
The Levenberg-Marquardt algorithm is a powerful tool for solving non-linear least squares problems. By combining the efficiency of Gauss-Newton with the robustness of gradient descent, it provides an adaptive approach to optimization that is both reliable and effective across a variety of domains.
