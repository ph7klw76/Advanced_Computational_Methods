
# Nonlinear Least Squares Fitting: A Mathematical Overview

Nonlinear least squares fitting is a method used to fit a model to data by minimizing the sum of the squares of the differences (residuals) between the observed data points and the corresponding points predicted by the model. It is an extension of linear least squares, but the key difference is that the relationship between the parameters and the dependent variable is **nonlinear**.

## 1. Problem Setup

Given a set of data points $(x_i, y_i)$ where $i = 1, 2, \dots, n$, the goal is to find the parameters $\theta = (\theta_1, \theta_2, \dots, \theta_p)$ of a model $f(x; \theta)$ such that the model fits the data as closely as possible.

The residual for each data point is:

$$
r_i(\theta) = y_i - f(x_i; \theta)
$$

The objective is to minimize the sum of the squares of the residuals:

$$
S(\theta) = \sum_{i=1}^{n} r_i(\theta)^2 = \sum_{i=1}^{n} \left[ y_i - f(x_i; \theta) \right]^2
$$

where $S(\theta)$ is the **objective function** or the **sum of squared residuals**.

## 2. Optimization Problem

The task of minimizing $S(\theta)$ is a **nonlinear optimization problem**, as $f(x; \theta)$ is a nonlinear function of the parameters $\theta$. Unlike linear least squares, this problem generally cannot be solved by a closed-form solution.

We can reformulate the problem as a minimization problem:

$$
\min_{\theta} S(\theta) = \min_{\theta} \sum_{i=1}^{n} \left[ y_i - f(x_i; \theta) \right]^2
$$

## 3. Gradient and Hessian

To minimize $S(\theta)$, iterative methods such as the Gauss-Newton algorithm or Levenberg-Marquardt algorithm are often used. These methods rely on the **gradient** and **Hessian** of the objective function.

- The **gradient** $\nabla S(\theta)$ is the vector of partial derivatives of $S(\theta)$ with respect to each parameter $\theta_j$:

$$
\frac{\partial S(\theta)}{\partial \theta_j} = -2 \sum_{i=1}^{n} r_i(\theta) \frac{\partial f(x_i; \theta)}{\partial \theta_j}
$$

- The **Hessian matrix** $H$ (a matrix of second derivatives) is used to measure the curvature of the objective function:

$$
H_{jk} = \frac{\partial^2 S(\theta)}{\partial \theta_j \partial \theta_k} = 2 \sum_{i=1}^{n} \left[ \frac{\partial f(x_i; \theta)}{\partial \theta_j} \frac{\partial f(x_i; \theta)}{\partial \theta_k} - r_i(\theta) \frac{\partial^2 f(x_i; \theta)}{\partial \theta_j \partial \theta_k} \right]
$$

- For **small residuals**, the second term in the Hessian can be neglected, and the Hessian is approximated as:

$$
H_{jk} \approx 2 \sum_{i=1}^{n} \frac{\partial f(x_i; \theta)}{\partial \theta_j} \frac{\partial f(x_i; \theta)}{\partial \theta_k}
$$

## 4. Iterative Methods

Since direct minimization of $S(\theta)$ is usually not feasible, **iterative methods** are employed.

- **Gauss-Newton Method**: This method approximates the Hessian matrix using the Jacobian matrix $J$ of partial derivatives of the residuals. The update rule is:

$$
\theta^{(k+1)} = \theta^{(k)} - (J^T J)^{-1} J^T r(\theta)
$$

where $J$ is the Jacobian of $f(x_i; \theta)$ with respect to $\theta$, and $r(\theta)$ is the residual vector.

- **Levenberg-Marquardt Algorithm**: This algorithm blends the Gauss-Newton method with gradient descent, adding a damping parameter $\lambda$ to control the step size:

$$
\theta^{(k+1)} = \theta^{(k)} - (J^T J + \lambda I)^{-1} J^T r(\theta)
$$

When $\lambda$ is large, the method behaves like gradient descent, and when $\lambda$ is small, it behaves like the Gauss-Newton method.

## 5. Convergence Criteria

The iterative method continues until one of the following conditions is met:

- The change in the parameters $\theta$ between iterations is smaller than a predefined threshold.
- The improvement in the sum of squared residuals $S(\theta)$ is smaller than a predefined tolerance.
- The number of iterations exceeds a specified limit.

## 6. Goodness of Fit and Statistical Properties

After fitting the model, it is essential to assess the **goodness of fit** and the reliability of the estimated parameters.

- **Residual Sum of Squares (RSS)**: The minimized value of $S(\theta)$ is called the residual sum of squares, which indicates how well the model fits the data.

$$
\text{RSS} = \sum_{i=1}^{n} \left[ y_i - f(x_i; \theta) \right]^2
$$

- **Covariance Matrix**: The covariance matrix of the parameter estimates is approximated as:

$$
\text{Cov}(\theta) \approx \sigma^2 (J^T J)^{-1}
$$

where $\sigma^2$ is the estimated variance of the errors, typically computed as:

$$
\sigma^2 = \frac{1}{n - p} \sum_{i=1}^{n} r_i(\theta)^2
$$

where $n$ is the number of data points, and $p$ is the number of parameters.

## 7. Example: Fitting an Exponential Model

Suppose we are fitting an exponential model $f(x; \theta) = \theta_1 e^{\theta_2 x}$ to data. The sum of squared residuals would be:

$$
S(\theta) = \sum_{i=1}^{n} \left[ y_i - \theta_1 e^{\theta_2 x_i} \right]^2
$$

To minimize this, we calculate the gradient and Hessian:

- **Gradient**:

$$
\frac{\partial S(\theta)}{\partial \theta_1} = -2 \sum_{i=1}^{n} \left[ y_i - \theta_1 e^{\theta_2 x_i} \right] e^{\theta_2 x_i}
$$

$$
\frac{\partial S(\theta)}{\partial \theta_2} = -2 \sum_{i=1}^{n} \left[ y_i - \theta_1 e^{\theta_2 x_i} \right] \theta_1 x_i e^{\theta_2 x_i}
$$

- **Hessian**: The second derivatives would involve more complex terms, but in practice, an iterative method (e.g., Levenberg-Marquardt) would be applied to solve this numerically.

## Conclusion

Nonlinear least squares fitting is mathematically more complex than linear least squares due to the nonlinearity of the model with respect to the parameters. The methods used to solve these problems rely on iterative algorithms such as Gauss-Newton and Levenberg-Marquardt, which approximate solutions using the gradient and Hessian of the objective function. The rigorous mathematical foundation involves optimization, matrix calculus, and numerical methods, all aimed at finding the parameters that best fit the model to the data.
