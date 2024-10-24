
# Introduction to Artificial Neural Networks (ANNs)

Artificial Neural Networks (ANNs) are computational models inspired by the human brain's network of neurons. ANNs are a crucial component of deep learning and are widely used in various fields such as image recognition, natural language processing, and even climate modeling. In this blog, we will explore the underlying mathematics of neural networks, focusing on the rigorous derivations and concepts that drive their operation.

## 1. The Structure of Artificial Neural Networks
An artificial neural network consists of layers of interconnected neurons:

**Input Layer**: Takes the input features (data points) into the network.\
**Hidden Layers**: These intermediate layers perform computations and feature transformations.\
**Output Layer**: Produces the final prediction or classification based on the processed data.

Each neuron in a layer performs a weighted sum of its inputs and passes this sum through an activation function. The neuron is mathematically represented as follows:

For a neuron $𝑗$ in a layer:
The equation for $z_j$ is given by:

$$
z_j = \sum_{i=1}^{n} w_{ji} x_i + b_j
$$

where:

- $x_i$ are the input features.
- $w_{ji}$ are the weights associated with the inputs to neuron $j$.
- $b_j$ is the bias term for neuron $j$.
- $z_j$ is the linear combination of the inputs and weights (pre-activation).

### 2. Forward Propagation

In forward propagation, the inputs are passed through the network, layer by layer, until an output is produced. For each neuron in a hidden or output layer, the process involves two steps:

1. **Linear combination of inputs** (as shown in the equation above).
2. **Activation function**: The neuron applies an activation function to the linear combination $z_j$ to introduce non-linearity:

$$
a_j = \sigma(z_j)
$$

where $\sigma$ is the activation function, and $a_j$ is the activated output of neuron $j$.

Let’s assume we have $L$ layers in the neural network. Denote the input layer as $X$ (with features $x_1, x_2, \dots, x_n$). For each layer $l = 1, 2, \dots, L$, the forward propagation process can be generalized as:

$$
Z^{(l)} = W^{(l)} A^{(l-1)} + b^{(l)}
$$

$$
A^{(l)} = \sigma(Z^{(l)})
$$

where:

- $W^{(l)}$ is the weight matrix for layer $l$.
- $A^{(l-1)}$ is the activated output from the previous layer.
- $b^{(l)}$ is the bias vector for layer $l$.
- $Z^{(l)}$ is the pre-activation output (linear combination).
- $A^{(l)}$ is the post-activation output for the current layer.

For the input layer, $A^{(0)} = X$, and for the output layer, $A^{(L)} = \hat{y}$, where $\hat{y}$ is the network’s final prediction.

### 3. Activation Functions

Activation functions introduce non-linearity into the network, enabling it to learn complex patterns. Common activation functions include:

- **Sigmoid**:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

  The sigmoid function squashes the output to a range between 0 and 1.

- **Tanh (Hyperbolic Tangent)**:

$$
\sigma(z) = \tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}
$$

  The output is squashed between -1 and 1, often used for symmetric outputs.

- **ReLU (Rectified Linear Unit)**:

$$
\sigma(z) = \max(0, z)
$$

  The ReLU function introduces sparsity by setting negative values to zero.

- **Softmax (for multi-class classification)**:

$$
\sigma(z_j) = \frac{e^{z_j}}{\sum_{k=1}^{K} e^{z_k}}
$$

  Softmax converts a vector of raw scores $z_j$ into a probability distribution, typically used in the output layer for classification tasks.

  ### 4. Loss Function and Error Computation

The loss function measures how well the neural network’s predictions match the true labels. The most common loss functions are:

- **Mean Squared Error (MSE)** (for regression tasks):

$$
L(\hat{y}, y) = \frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)^2
$$

where $\hat{y}_i$ is the predicted value, and $y_i$ is the true value.

- **Cross-Entropy Loss** (for classification tasks):

$$
L(\hat{y}, y) = -\sum_{i=1}^{n} y_i \log(\hat{y}_i)
$$

where $y_i$ are the true class labels (one-hot encoded), and $\hat{y}_i$ are the predicted probabilities from the network.

The objective of training is to minimize the loss function with respect to the network's parameters (weights and biases). This is where backpropagation and gradient descent come into play.

### 5. Backpropagation and Gradient Descent

Backpropagation and Gradient Descent are key techniques used to train Artificial Neural Networks (ANNs). The process involves calculating the gradients of the loss function with respect to the network parameters (weights and biases) and updating these parameters using gradient descent to minimize the loss function. In this section, we dive into the mathematical foundations of backpropagation, starting from the chain rule of calculus to derive the gradients and exploring how these gradients are used in gradient descent.

#### Overview of Backpropagation

The goal of backpropagation is to efficiently compute the gradients of the loss function with respect to all weights and biases in the neural network. This is done by propagating the error from the output layer back through the network to the input layer. The computed gradients are then used to update the parameters during gradient descent.

In a neural network with multiple layers, we compute the partial derivatives of the loss function $L$ with respect to each weight and bias in every layer. The key idea is to use the chain rule to compute these derivatives recursively.

Let’s assume we are dealing with a feedforward neural network with $L$ layers. The notations used are as follows:

- $W^{(l)}$: Weight matrix for layer $l$.
- $b^{(l)}$: Bias vector for layer $l$.
- $A^{(l)}$: Output of layer $l$ after applying the activation function (also called activations).
- $Z^{(l)}$: Pre-activation values (linear combination of the previous layer's activations).
- $\sigma$: Activation function.
- $L$: Loss function, typically calculated over all training examples.

#### 5.1 Forward Propagation Recap

In forward propagation, for a given input $X$, we compute the output of the neural network step by step, layer by layer.

For the input to the $l$-th layer, we have the following two steps:

- **Linear Combination**:

$$
Z^{(l)} = W^{(l)} A^{(l-1)} + b^{(l)}
$$

where $A^{(l-1)}$ is the activation from the previous layer (with $A^{(0)} = X$).

- **Activation**:

$$
A^{(l)} = \sigma(Z^{(l)})
$$

where $\sigma$ is the activation function applied element-wise to $Z^{(l)}$.

For the output layer $L$, we have the prediction:

$$
\hat{y} = A^{(L)} = \sigma(Z^{(L)})
$$

#### 5.2 The Loss Function

The loss function measures how far the network's predictions $\hat{y}$ are from the true labels $y$. Common loss functions include the mean squared error for regression and cross-entropy for classification.

- For **mean squared error**:

$$
L(\hat{y}, y) = \frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)^2
$$

where $\hat{y}_i$ is the predicted output for the $i$-th sample, and $y_i$ is the true label.

- For **cross-entropy loss** (for multi-class classification):

$$
L(\hat{y}, y) = -\sum_{i=1}^{n} y_i \log(\hat{y}_i)
$$

where $y_i$ is the true class label (often one-hot encoded), and $\hat{y}_i$ is the predicted probability for class $i$.

The objective of training is to minimize this loss function $L$ with respect to the network parameters (weights $W^{(l)}$ and biases $b^{(l)}$).

#### 5.3 Gradient Descent

Gradient Descent is an optimization algorithm used to minimize the loss function by updating the network's parameters in the direction of the negative gradient. The update rule for a parameter $\theta$ (which can be a weight or bias) is:

$$
\theta \leftarrow \theta - \eta \frac{\partial L}{\partial \theta}
$$

where:

- $\eta$ is the learning rate, a small positive constant that controls the step size.
- $\frac{\partial L}{\partial \theta}$ is the gradient of the loss function with respect to $\theta$.

In a neural network, gradient descent is applied to update all weights $W^{(l)}$ and biases $b^{(l)}$ across all layers $l=1, \dots, L$.

#### 5.4 Deriving Backpropagation Using the Chain Rule

To derive the gradients needed for gradient descent, we use the chain rule of calculus. The goal is to compute:

$$
\frac{\partial L}{\partial W^{(l)}}, \quad \frac{\partial L}{\partial b^{(l)}}
$$

##### 5.4.1 Output Layer Gradients

Let's start with the output layer, $l = L$, where we compute the error with respect to the network's output $A^{(L)}$.

Define the error term at the output layer as:

$$
\delta^{(L)} = \frac{\partial L}{\partial A^{(L)}} \odot \sigma'(Z^{(L)})
$$

where:

- $\delta^{(L)}$ is the error term for the output layer.
- $\frac{\partial L}{\partial A^{(L)}}$ is the derivative of the loss function with respect to the activations of the output layer.
- $\sigma'(Z^{(L)})$ is the derivative of the activation function with respect to the pre-activation values $Z^{(L)}$.
- $\odot$ denotes element-wise multiplication.

For example, if we are using a cross-entropy loss combined with a softmax activation function in the output layer, $\frac{\partial L}{\partial A^{(L)}} = A^{(L)} - y$, where $y$ is the true label.

##### 5.4.2 Hidden Layer Gradients

For each hidden layer $l$, we propagate the error backward using the chain rule. The error at layer $l$ is computed by propagating the error from the next layer $l+1$ as follows:

$$
\delta^{(l)} = (W^{(l+1)})^T \delta^{(l+1)} \odot \sigma'(Z^{(l)})
$$

##### 5.4.3 Gradients with Respect to Weights and Biases

Once the error terms $\delta^{(l)}$ are computed for each layer, we can calculate the gradients of the loss function with respect to the weights and biases.

- For the weights $W^{(l)}$ of layer $l$, the gradient is:

$$
\frac{\partial L}{\partial W^{(l)}} = \delta^{(l)} (A^{(l-1)})^T
$$

where $A^{(l-1)}$ are the activations from the previous layer.

- For the biases $b^{(l)}$, the gradient is:

$$
\frac{\partial L}{\partial b^{(l)}} = \delta^{(l)}
$$

These gradients are used to update the weights and biases via gradient descent.

#### 5.5 Gradient Descent: Stochastic, Mini-Batch, and Batch

- **Stochastic Gradient Descent (SGD)**: The weights are updated after computing the gradient from a single training example:

$$
\theta \leftarrow \theta - \eta \frac{\partial L_i}{\partial \theta}
$$

where $L_i$ is the loss for the $i$-th training example.

- **Mini-Batch Gradient Descent**: The weights are updated after computing the gradient from a small batch of training examples:

$$
\theta \leftarrow \theta - \eta \frac{1}{m} \sum_{i=1}^{m} \frac{\partial L_i}{\partial \theta}
$$

where $m$ is the batch size.

#### 5.6 Optimizing Gradient Descent: Momentum, RMSProp, and Adam

In practice, vanilla gradient descent can suffer from issues like slow convergence or getting stuck in local minima. To address these issues, several optimizers have been developed to improve gradient descent:

- **Momentum**: Momentum accelerates gradient descent by adding a fraction of the previous update to the current update:

$$
v_t = \gamma v_{t-1} + \eta \nabla L(\theta)
$$

$$
\theta \leftarrow \theta - v_t
$$

where $v_t$ is the velocity, and $\gamma$ is the momentum term.

- **RMSProp**: RMSProp adapts the learning rate for each parameter by dividing the gradient by the root of the squared gradients' running average:

$$
E[g^2]_t =
$$

$$
\beta E[g^2]_{t-1} + (1-\beta) g_t^2
$$

$$
\theta \leftarrow \theta - \frac{\eta}{\sqrt{E[g^2]_t + \epsilon}} g_t
$$

where $\beta$ is the decay term, and $g_t$ is the gradient at step $t$.

- **Adam (Adaptive Moment Estimation)**: Adam combines the ideas of momentum and RMSProp by maintaining both an exponentially decaying average of past gradients and squared gradients:

$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t
$$

$$
v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
$$

$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}
$$

$$
\theta \leftarrow \theta - \frac{\eta \hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

Adam provides adaptive learning rates for each parameter and is widely used in modern neural network training.
### 6. Example: Multi-Layer Perceptron (MLP)

A Multi-Layer Perceptron (MLP) is a fully connected feedforward neural network, meaning each neuron in a layer is connected to every neuron in the next layer. Here's an example with two hidden layers:

- **Input Layer**: Takes a vector $X \in \mathbb{R}^n$.
- **First Hidden Layer**: With $m_1$ neurons.

$$ 
Z^{(1)} = W^{(1)} X + b^{(1)}, \quad A^{(1)} = \sigma(Z^{(1)})
$$

- **Second Hidden Layer**: With $m_2$ neurons.

$$ 
Z^{(2)} = W^{(2)} A^{(1)} + b^{(2)}, \quad A^{(2)} = \sigma(Z^{(2)})
$$

- **Output Layer**: With $k$ outputs (for multi-class classification).

$$ 
Z^{(L)} = W^{(L)} A^{(L-1)} + b^{(L)}, \quad \hat{y} = \sigma(Z^{(L)})
$$

#### Training an MLP involves:

1. **Forward propagation** to compute the predictions.
2. **Loss calculation** to assess the model’s performance.
3. **Backpropagation** to calculate the gradients.
4. **Gradient descent** to update the weights and biases iteratively.


