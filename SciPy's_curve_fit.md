# How SciPy's `curve_fit` Implements Nonlinear Least Squares Fitting

`curve_fit` in `SciPy` is a powerful function that implements nonlinear least squares fitting, commonly using the **Levenberg-Marquardt** algorithm by default. Here's a detailed breakdown of how `curve_fit` works, following the nonlinear least squares process described earlier.

## 1. Problem Setup

When using `curve_fit`, you need to provide:
- A **model function** $f(x; \theta)$, which represents the relationship between the independent variable $x$ and the parameters $\theta$.
- A set of **data points** $(x_i, y_i)$, where $x_i$ is the independent variable and $y_i$ is the observed dependent variable.
- An **initial guess** for the parameters $\theta_0$.

In `SciPy`:
- The model function is passed as the first argument.
- The data points $(x_i, y_i)$ are passed as the second and third arguments.
- Optional initial guesses for the parameters can be passed via the `p0` argument.

## 2. Objective Function

`curve_fit` aims to minimize the sum of squared residuals:

$$
S(\theta) = \sum_{i=1}^{n} \left[ y_i - f(x_i; \theta) \right]^2
$$

Where:
- $f(x_i; \theta)$ is the model function.
- $y_i$ are the observed values.
- $\theta$ are the model parameters.

The goal is to find the parameters $\theta$ that minimize this sum of squared residuals.

## 3. Iterative Methods: Levenberg-Marquardt

By default, `curve_fit` uses the **Levenberg-Marquardt algorithm** for optimization, which is a hybrid method blending **Gauss-Newton** and **gradient descent**. The update rule is:

$$
\theta^{(k+1)} = \theta^{(k)} - (J^T J + \lambda I)^{-1} J^T r(\theta)
$$

Where:
- $J$ is the **Jacobian** matrix of partial derivatives of the residuals with respect to each parameter.
- $r(\theta)$ is the vector of residuals $r_i(\theta) = y_i - f(x_i; \theta)$.
- $\lambda$ is a damping factor that is adjusted during the iterations.

### Damping Factor

- If a step results in a **lower** residual sum of squares, $\lambda$ is reduced, making the algorithm behave more like Gauss-Newton.
- If a step **increases** the residual sum of squares, $\lambda$ is increased, making the algorithm behave more like gradient descent (slower, more cautious steps).

## 4. Jacobian Calculation

The **Jacobian** $J$ is the matrix of first derivatives of the model function with respect to the parameters $\theta$. 

- By default, the Jacobian is estimated **numerically** by perturbing each parameter and observing the change in residuals (finite-difference method). This is computationally expensive but requires no analytical expressions.
- Alternatively, you can **supply an analytical Jacobian** (via the `jac` argument) for better efficiency.

## 5. Covariance Matrix and Parameter Uncertainty

After convergence, `curve_fit` returns:
- The optimized parameters $\hat{\theta}$ that minimize the sum of squared residuals.
- The **covariance matrix** of the parameters, which is given by:

$$
\text{Cov}(\theta) \approx \sigma^2 (J^T J)^{-1}
$$

Where $\sigma^2$ is the estimated variance of the residuals:

$$
\sigma^2 = \frac{1}{n - p} \sum_{i=1}^{n} \left( y_i - f(x_i; \theta) \right)^2
$$

Here:
- $n$ is the number of data points.
- $p$ is the number of fitted parameters.

The covariance matrix is crucial for estimating the **standard error** and **confidence intervals** for the parameters.

## 6. Convergence Criteria

The iterative fitting process in `curve_fit` stops when one of the following criteria is met:
- The **relative change** in the parameters $\theta$ between iterations is smaller than a predefined threshold.
- The **improvement** in the sum of squared residuals $S(\theta)$ is smaller than a predefined tolerance.
- The number of iterations exceeds a specified limit (controlled with the `maxfev` argument).

## 7. Handling Bounds and Other Methods

Although the **Levenberg-Marquardt** algorithm is the default, `curve_fit` can handle **bounded constraints** on the parameters by using a **Trust Region Reflective** algorithm. If you specify bounds on the parameters, `curve_fit` automatically switches to this method instead.


## Example of `curve_fit` in Action



Here's an example of how `curve_fit` works in practice:

```python
import numpy as np
from scipy.optimize import curve_fit

# Define the model function
def model(x, a, b):
    return a * np.exp(b * x)

# Example data
x_data = np.linspace(0, 4, 50)
y_data = model(x_data, 2.5, 1.3) + 0.2 * np.random.normal(size=len(x_data))

# Perform curve fitting
initial_guess = [1, 1]  # Initial guess for parameters a and b
params, covariance = curve_fit(model, x_data, y_data, p0=initial_guess)

# params will contain the optimized parameters a and b
# covariance contains the covariance matrix of the parameters
```
## Example Explained

In this example:

- The model is $f(x; \theta) = a e^{b x}$.
- We generate synthetic data with some noise.
- `curve_fit` finds the optimal values for $a$ and $b$ and returns them along with the covariance matrix, which can be used for error estimation.

### Key Steps Summarized:

1. **Initial Guess**: Provide an initial guess for the parameters.
   
2. **Model Function**: Define the model relating the independent variable $x$ and the dependent variable $y$. In this case, the model is $f(x; \theta) = a e^{b x}$.
   
3. **Jacobian & Residual Calculation**: `curve_fit` computes the Jacobian matrix and residuals at each iteration. The residuals are the differences between the observed data $y_i$ and the model prediction $f(x_i; \theta)$, and the Jacobian is the matrix of partial derivatives of these residuals with respect to the parameters.
   
4. **Levenberg-Marquardt Algorithm**: The parameters are updated iteratively using the **Levenberg-Marquardt algorithm**, which blends Gauss-Newton with gradient descent. The update rule is:

   $\theta^{(k+1)} = \theta^{(k)} - \left( J^T J + \lambda I \right)^{-1} J^T r(\theta)$

   where $J$ is the Jacobian matrix and $r(\theta)$ is the residual vector.
   
6. **Convergence**: The process stops when the change in parameters between iterations is smaller than a predefined threshold or when the improvement in the sum of squared residuals is sufficiently small.

7. **Parameter Estimation**: Returns the best-fit parameters and the **covariance matrix**, which is used to estimate the uncertainty in the parameter values. The covariance matrix is computed as:

   $\text{Cov}(\theta) \approx \sigma^2 (J^T J)^{-1}$

   where $\sigma^2$ is the estimated variance of the residuals, typically given by:

   $\sigma^2 = \frac{1}{n - p} \sum_{i=1}^{n} \left( y_i - f(x_i; \theta) \right)^2$


## Conclusion

The `curve_fit` function from `SciPy` automates **nonlinear least squares fitting** using iterative optimization methods. It employs the **Levenberg-Marquardt algorithm** to minimize the sum of squared residuals, producing a solution that estimates the model parameters and computes the covariance matrix, making it a versatile tool for nonlinear regression tasks.

