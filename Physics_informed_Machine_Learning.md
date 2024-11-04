# Introduction

Traditional numerical methods for solving partial differential equations (PDEs)—such as finite difference, finite element, and spectral methods—rely on discretizing the domain into a mesh or grid. While effective, these methods can become computationally intensive, especially for high-dimensional problems or complex geometries. Additionally, generating meshes for irregular domains can be challenging.

Physics-Informed Neural Networks (PINNs) offer an alternative by embedding the physical laws governing a system directly into the structure of a neural network. This approach allows for:

- **Mesh-free computations**: Eliminating the need for domain discretization.
- **Integration of data and physics**: Combining observational data with known physical laws.
- **Solving inverse problems**: Estimating unknown parameters or inputs in the governing equations.

This guide aims to provide a detailed understanding of PINNs and inverse PINNs, explaining the mathematical concepts and equations involved.

# Mathematical Foundations of PINNs

## Partial Differential Equations

A partial differential equation (PDE) is an equation involving an unknown function $u(x)$ of multiple variables and its partial derivatives. PDEs are fundamental in describing various physical phenomena such as heat conduction, fluid flow, and electromagnetic fields.

A general form of a PDE is:

$$
N_x [u(x)] = f(x), \quad x \in \Omega \subset \mathbb{R}^n,
$$

where:
- $N_x$ is a differential operator acting on $u(x)$.
- $u(x)$ is the unknown solution we seek.
- $f(x)$ is a known source term.
- $x$ represents the spatial (and possibly temporal) variables.
- $\Omega$ is the domain of interest in $n$-dimensional space.

### Example: The One-Dimensional Heat Equation

The one-dimensional heat equation is a PDE given by:

$$
\frac{\partial u}{\partial t} = \kappa \frac{\partial^2 u}{\partial x^2}, \quad x \in [0, L], \, t > 0,
$$

where $\kappa$ is the thermal diffusivity.

## Neural Network Approximation of Solutions

In PINNs, we approximate the unknown solution $u(x)$ using a neural network $u_\theta(x)$, where $\theta$ represents the network parameters (weights and biases).

### Key Concepts

- **Function Approximation**: Neural networks are universal function approximators, capable of approximating any continuous function under certain conditions.
- **Automatic Differentiation**: A technique that allows exact computation of derivatives of functions represented by neural networks, essential for evaluating the PDE residuals.

The goal is to find parameters $\theta$ such that $u_\theta(x)$ satisfies:
1. The governing PDE.
2. Boundary and initial conditions.

# Formulating the PINN Loss Function

The loss function in PINNs is designed to penalize deviations from the PDE and the boundary/initial conditions. It guides the neural network during training to find a solution that adheres to the physical laws.

## Physics-Informed Loss

The physics-informed loss quantifies how well the neural network solution $u_\theta(x)$ satisfies the PDE at a set of collocation points $\{x_i\}_{i=1}^{N_f}$ within the domain $\Omega$.

### Definition

$$
L_{\text{PDE}}(\theta) = \frac{1}{N_f} \sum_{i=1}^{N_f} \left| N_{x_i}[u_\theta(x_i)] - f(x_i) \right|^2,
$$

where:
- $N_f$ is the number of collocation points.
- $N_{x_i}[u_\theta(x_i)]$ is the PDE operator evaluated at $x_i$ using the neural network approximation.
- $f(x_i)$ is the source term evaluated at $x_i$.

### Explanation

For each collocation point $x_i$, we compute the residual $r_i = N_{x_i}[u_\theta(x_i)] - f(x_i)$. The residual measures the discrepancy between the neural network's output and what the PDE dictates at $x_i$. The physics-informed loss aggregates these discrepancies over all collocation points.

## Boundary and Initial Conditions

Boundary and initial conditions are crucial for well-posedness of PDEs. In PINNs, we enforce these conditions through additional terms in the loss function.

### Definition

$$
L_{\text{BC}}(\theta) = \frac{1}{N_b} \sum_{i=1}^{N_b} \left| B_{x_i}[u_\theta(x_i)] - g(x_i) \right|^2,
$$

where:
- $N_b$ is the number of boundary points.
- $B_{x_i}[u_\theta(x_i)]$ represents the boundary condition operator applied to $u_\theta(x_i)$.
- $g(x_i)$ is the prescribed boundary condition value at $x_i$.

### Explanation

This term penalizes deviations from the boundary conditions at specified points on the boundary $\partial \Omega$.

#### Example of Boundary Conditions

- **Dirichlet Condition**: $u(x) = g(x)$ on $\partial \Omega$.
- **Neumann Condition**: $\frac{\partial u}{\partial n} = h(x)$ on $\partial \Omega$, where $n$ is the outward normal.

## Total Loss Function

The total loss function combines the physics-informed loss and the boundary condition loss, often with weighting factors to balance their contributions.

### Definition

$$
L(\theta) = L_{\text{PDE}}(\theta) + \lambda L_{\text{BC}}(\theta),
$$

where $\lambda$ is a hyperparameter that controls the relative importance of satisfying the PDE versus the boundary conditions.

### Explanation

By minimizing $L(\theta)$, we aim to find a neural network $u_\theta(x)$ that approximately satisfies both the PDE and the boundary conditions across the domain.

# Training Physics-Informed Neural Networks

Training a PINN involves optimizing the neural network parameters $\theta$ to minimize the total loss function $L(\theta)$.

### Training Procedure

1. **Initialize** the neural network parameters $\theta$ randomly or based on prior knowledge.
2. **Sample Collocation Points**:
   - Generate $N_f$ collocation points $\{x_i\}_{i=1}^{N_f}$ in the domain $\Omega$.
   - Generate $N_b$ boundary points $\{x_i\}_{i=1}^{N_b}$ on the boundary $\partial \Omega$.
3. **Compute Residuals**:
   - Evaluate the PDE residuals at the collocation points using automatic differentiation.
   - Evaluate the boundary condition residuals at the boundary points.
4. **Compute Total Loss**:
   - Calculate $L_{\text{PDE}}(\theta)$ and $L_{\text{BC}}(\theta)$.
   - Sum them to obtain $L(\theta)$.
5. **Optimize Parameters**:
   - Use optimization algorithms (e.g., stochastic gradient descent, Adam optimizer) to update $\theta$ in the direction that minimizes $L(\theta)$.
6. **Iterate**:
   - Repeat steps 3-5 until convergence criteria are met (e.g., loss below a threshold, maximum number of iterations).

### Important Notes

- **Automatic Differentiation**: Essential for computing derivatives of $u_\theta(x)$ with respect to $x$, needed in $N_{x_i}[u_\theta(x_i)]$.
- **Hyperparameters**: The choice of $\lambda$, learning rate, and network architecture (number of layers, neurons) can significantly affect performance.
- **Convergence**: Monitoring the loss components separately can help diagnose issues during training.

  # Inverse Problems with PINNs

## Understanding Inverse Problems

An inverse problem seeks to determine unknown parameters or inputs in a PDE from observed data. This contrasts with forward problems, where the parameters are known, and the solution $u(x)$ is sought.

### Examples of Inverse Problems
- **Estimating material properties** (e.g., thermal conductivity, diffusion coefficients).
- **Identifying sources or sinks** in a domain.
- **Determining initial conditions** from later observations.

## Inverse PINN Architecture

To tackle inverse problems, we extend the PINN framework to include unknown parameters as additional variables to be learned during training.

### Approach
1. **Augment Neural Network**: Include unknown parameters $\lambda$ as trainable variables alongside $\theta$.
2. **Modify Loss Function**: Incorporate data mismatch terms to enforce agreement with observed data.
3. **Simultaneous Optimization**: Optimize both $\theta$ and $\lambda$ to minimize the total loss.

### Mathematical Formulation
Suppose we have a PDE with unknown parameter $\lambda$:

$$
N_x [u(x); \lambda] = f(x), \quad x \in \Omega,
$$

with boundary conditions:

$$
B_x [u(x)] = g(x), \quad x \in \partial \Omega.
$$

### Loss Function Components

1. **Physics-Informed Loss**:

    $$
    L_{\text{PDE}}(\theta, \lambda) = \frac{1}{N_f} \sum_{i=1}^{N_f} \left| N_{x_i}[u_\theta(x_i); \lambda] - f(x_i) \right|^2.
    $$

2. **Boundary Condition Loss**:

    $$
    L_{\text{BC}}(\theta) = \frac{1}{N_b} \sum_{i=1}^{N_b} \left| B_{x_i}[u_\theta(x_i)] - g(x_i) \right|^2.
    $$

3. **Data Loss**:

    $$
    L_{\text{Data}}(\theta) = \frac{1}{N_d} \sum_{i=1}^{N_d} \left| u_\theta(x_i) - u_{\text{obs}}(x_i) \right|^2,
    $$

    where $\{(x_i, u_{\text{obs}}(x_i))\}_{i=1}^{N_d}$ are observed data points.

4. **Total Loss Function**:

    $$
    L(\theta, \lambda) = L_{\text{PDE}}(\theta, \lambda) + \lambda_1 L_{\text{BC}}(\theta) + \lambda_2 L_{\text{Data}}(\theta),
    $$

    with $\lambda_1$ and $\lambda_2$ as weighting factors.

### Optimization Objective

$$
\min_{\theta, \lambda} \, L(\theta, \lambda).
$$

## Mathematical Details of Inverse PINNs

### Incorporating Unknown Parameters

In inverse PINNs, unknown parameters $\lambda$ become part of the optimization variables. They can be scalar values, vectors, or even functions, depending on the problem.

- **Scalar Parameter**: Estimating a constant diffusion coefficient $D$.
- **Spatially Varying Parameter**: Estimating $D(x)$ that varies with position.

### Regularization

To ensure physical plausibility and prevent overfitting to noisy data, regularization terms can be added to the loss function.

**Regularization Term**:

$$
L_{\text{Reg}}(\lambda) = \alpha \|\lambda - \lambda_0\|^2,
$$

where:
- $\lambda_0$ is a prior estimate or expected value of $\lambda$.
- $\alpha$ is the regularization coefficient controlling the strength of the penalty.

**Total Loss Function with Regularization**:

$$
L(\theta, \lambda) = L_{\text{PDE}}(\theta, \lambda) + \lambda_1 L_{\text{BC}}(\theta) + \lambda_2 L_{\text{Data}}(\theta) + L_{\text{Reg}}(\lambda).
$$

### Training Procedure

1. **Initialize** $\theta$ and $\lambda$.
2. **Sample** collocation, boundary, and data points.
3. **Compute Residuals**:
    - PDE residuals using $\lambda$.
    - Boundary condition residuals.
    - Data residuals comparing $u_\theta(x_i)$ to $u_{\text{obs}}(x_i)$.
4. **Compute Total Loss**: Include regularization if applicable.
5. **Optimize Parameters**: Update $\theta$ and $\lambda$ using optimization algorithms.
6. **Iterate** until convergence.

### Challenges
- **Identifiability**: Ensuring the problem is well-posed and that the data contain enough information to estimate $\lambda$.
- **Optimization Stability**: Jointly optimizing $\theta$ and $\lambda$ can be sensitive; careful tuning of hyperparameters is necessary.

## Applications of PINNs and Inverse PINNs

### Forward Problems

1. **Heat Equation**:
    - **Problem**: Solve $\frac{\partial u}{\partial t} - \kappa \nabla^2 u = 0$ with known thermal diffusivity $\kappa$.
    - **Application**: Modeling temperature distribution over time in a solid object.

2. **Wave Equation**:
    - **Problem**: Solve $\frac{\partial^2 u}{\partial t^2} - c^2 \nabla^2 u = 0$ with known wave speed $c$.
    - **Application**: Simulating vibrations in a string or pressure waves.

### Inverse Problems

1. **Parameter Estimation**:
    - **Problem**: Estimate $\kappa$ in the heat equation from observed temperature data.
    - **Application**: Determining material properties in thermography.

2. **Source Identification**:
    - **Problem**: Find unknown source term $f(x)$ in $N_x [u(x)] = f(x)$ using observations of $u(x)$.
    - **Application**: Locating pollutant sources in environmental modeling.

## Example: Estimating Diffusion Coefficient

Consider the one-dimensional diffusion equation with unknown diffusion coefficient $D$:

$$
\frac{\partial u}{\partial t} = D \frac{\partial^2 u}{\partial x^2}, \quad x \in [0, L], \, t \in [0, T].
$$

### Inverse PINN Setup

- **Neural Network Outputs**: $u_\theta(x, t)$, approximating the solution $u(x, t)$.
- **Unknown Parameter**: $D$, treated as a trainable variable.

### Loss Function

$$
L(\theta, D) = L_{\text{PDE}}(\theta, D) + \lambda_1 L_{\text{BC}}(\theta) + \lambda_2 L_{\text{Data}}(\theta).
$$

- **Data**: Observations $\{(x_i, t_i, u_{\text{obs}}(x_i, t_i))\}_{i=1}^{N_d}$.

### Goal

Simultaneously learn $u_\theta(x, t)$ and estimate $D$ by minimizing $L$.

# Physics-Informed Neural Networks (PINNs) for Modeling an Underdamped Harmonic Oscillator

The application of Physics-Informed Neural Networks (PINNs) has emerged as a powerful method for solving complex, physics-driven problems. PINNs leverage the physics laws, represented by partial differential equations (PDEs), in neural network training. This blog delves into implementing PINNs in PyTorch for modeling an underdamped harmonic oscillator. We explore the theoretical underpinnings, key implementation details, and the advantages of using PINNs in such problems.

## Introduction to the Underdamped Harmonic Oscillator

The underdamped harmonic oscillator represents systems where the oscillatory motion gradually diminishes over time due to a damping force that resists the motion. This system can be described by a second-order differential equation:

$$
\ddot{u}(t) + 2d \dot{u}(t) + w_0^2 u(t) = 0,
$$

where:
- $u(t)$ is the displacement at time $t$,
- $d$ is the damping coefficient,
- $w_0$ is the natural frequency of the system.

### Solution to the Differential Equation

The solution to this equation, under underdamping conditions $d < w_0$, is:

$$
u(t) = A e^{-d t} \cos(w t + \phi),
$$

where:
- $w = \sqrt{w_0^2 - d^2}$ is the damped frequency,
- $A$ is the amplitude, and
- $\phi$ is the phase shift.

## Analytical Solution: Implementing the Exact Solution in PyTorch

In the first step of the code, we define the analytical solution to the underdamped oscillator. This exact solution serves as a benchmark for evaluating the accuracy of the PINN model.

```python
def exact_solution(d, w0, t):
    assert d < w0
    w = np.sqrt(w0**2 - d**2)
    phi = np.arctan(-d / w)
    A = 1 / (2 * np.cos(phi))
    cos = torch.cos(phi + w * t)
    exp = torch.exp(-d * t)
    u = exp * 2 * A * cos
    return u
```

This function utilizes trigonometric and exponential functions to simulate the decaying oscillation in PyTorch, generating synthetic data for comparison.

## Defining the Physics-Informed Neural Network (PINN)

To solve the differential equation using a neural network, we define a fully connected neural network (FCN) class with hidden layers and activation functions. This network serves as the core of the PINN, approximating the solution to the differential equation.

```python
import torch
import torch.nn as nn

class FCN(nn.Module):
    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS):
        super().__init__()
        activation = nn.Tanh
        self.fcs = nn.Sequential(
            nn.Linear(N_INPUT, N_HIDDEN),
            activation()
        )
        self.fch = nn.Sequential(
            *[nn.Sequential(
                nn.Linear(N_HIDDEN, N_HIDDEN),
                activation()
            ) for _ in range(N_LAYERS - 1)]
        )
        self.fce = nn.Linear(N_HIDDEN, N_OUTPUT)

    def forward(self, x):
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)
        return x
```

In this class:

- **`N_INPUT`**, **`N_OUTPUT`**, **`N_HIDDEN`**, and **`N_LAYERS`** represent the input dimension, output dimension, hidden layer width, and number of hidden layers, respectively.
- We use `Tanh` activations due to their suitability for approximating smooth functions.

## Setting Up Boundary and Physics Constraints

The success of PINNs lies in integrating boundary conditions and the governing physics equations into the training loss. The model calculates two types of losses:

1. **Boundary Loss**: Enforcing initial conditions by penalizing deviations from the boundary values.
2. **Physics Loss**: Minimizing the residuals of the differential equation across a chosen domain.

### Code for Boundary and Physics Points

```python
# Define boundary and physics points
t_boundary = torch.tensor(0.).view(-1, 1).requires_grad_(True)
t_physics = torch.linspace(0, 1, 30).view(-1, 1).requires_grad_(True)
```

This setup defines points over which to calculate boundary and physics losses, ensuring the solution respects initial conditions and physics laws.

## Training Process: Optimizing the Loss Function

The PINN loss combines boundary and physics losses with tunable parameters $\lambda_1$ and $\lambda_2$ to balance their contributions:

```python
lambda1, lambda2 = 1e-1, 1e-4

# Compute boundary loss
u = pinn(t_boundary)
loss1 = (torch.squeeze(u) - 1) ** 2
dudt = torch.autograd.grad(u, t_boundary, torch.ones_like(u), create_graph=True)[0]
loss2 = (torch.squeeze(dudt) - 0) ** 2

# Compute physics loss
u = pinn(t_physics)
dudt = torch.autograd.grad(u, t_physics, torch.ones_like(u), create_graph=True)[0]
d2udt2 = torch.autograd.grad(dudt, t_physics, torch.ones_like(dudt), create_graph=True)[0]
loss3 = torch.mean((d2udt2 + mu * dudt + k * u) ** 2)

# Backpropagate joint loss, take optimizer step
loss = loss1 + lambda1 * loss2 + lambda2 * loss3
loss.backward()
optimizer.step()
```

### The Loss Terms

- **`loss1`**: Enforces the initial position condition.
- **`loss2`**: Enforces the initial velocity condition.
- **`loss3`**: Satisfies the oscillator’s differential equation across the entire domain.

## Adding Noise to Observational Data

To improve robustness, the code introduces noise to synthetic data points, simulating real-world conditions where observational data may include measurement errors.

```python
t_obs = torch.rand(40).view(-1, 1)
u_obs = exact_solution(d, w0, t_obs) + 0.04 * torch.randn_like(t_obs)
```

Adding noise replicates measurement inaccuracies, providing a realistic training scenario for PINNs, which are tasked with balancing noisy observational data with governing physical laws.

## Adaptive Parameter Training with Learnable $\mu$

The code also introduces $\mu$ as a learnable parameter, allowing it to be updated dynamically based on observed data. This technique enables the model to fine-tune the physical parameter $\mu = 2d$ as it learns, making the training process more adaptive and responsive to data.

```python
mu = torch.nn.Parameter(torch.zeros(1, requires_grad=True))
optimizer = torch.optim.Adam(list(pinn.parameters()) + [mu], lr=1e-3)
```

This adaptive approach enables the network to estimate unknown physical constants, making PINNs ideal for scenarios where some system parameters are uncertain or variable.

## Enhanced Ansatz Formulation with Additional Parameters

Further flexibility is introduced with parameters $a$ and $b$ in the ansatz solution. These parameters enhance the solution's adaptability, allowing the model to accommodate more complex boundary and initial conditions.

```python
a = torch.nn.Parameter(70 * torch.ones(1, requires_grad=True))
b = torch.nn.Parameter(torch.ones(1, requires_grad=True))
optimizer = torch.optim.Adam(list(pinn.parameters()) + [a, b], lr=1e-3)
```








