
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

```python
# Import necessary libraries
import numpy as np
import pandas as pd

# Matminer for materials data and featurization
from matminer.datasets import load_dataset
from matminer.featurizers.composition import ElementProperty

# Scikit-learn for data splitting and preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

# TensorFlow Keras for building neural network models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

# Load the dielectric constant dataset from matminer
df = load_dataset('dielectric_constant')
print("Dataset loaded with {} entries.".format(len(df)))

# Extract composition from the structure
df['composition'] = df['structure'].apply(lambda x: x.composition)

# Initialize the ElementProperty featurizer with "magpie" preset
ep_featurizer = ElementProperty.from_preset(preset_name="magpie")

# Featurize the compositions
df = ep_featurizer.featurize_dataframe(df, col_id='composition', ignore_errors=True)
print("Featurization complete. Number of features: {}.".format(len(ep_featurizer.feature_labels())))

# Define features (X) and target variable (y)
X = df[ep_featurizer.feature_labels()]
y = df['n']  # Refractive index

# Drop any rows with missing values
X = X.dropna()
y = y.loc[X.index]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("Data split into training and testing sets.")

# Standardize the feature data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Feature scaling complete.")

# Build the artificial neural network model
model = Sequential()
model.add(Dense(128, input_dim=X_train_scaled.shape[1], activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))  # Output layer for regression
print("Neural network model constructed.")

# Compile the model
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model
history = model.fit(
    X_train_scaled, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.1,
    verbose=1  # Set to 1 to see training progress
)
print("Model training complete.")

# Evaluate the model on the test set
y_pred = model.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
print("Model evaluation complete.")
print(f"Test Mean Absolute Error (MAE): {mae:.4f}")

# Plotting the predicted vs actual values
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel('Actual Refractive Index')
plt.ylabel('Predicted Refractive Index')
plt.title('Actual vs Predicted Refractive Index')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.show()
```

This Python code demonstrates a real-world application of Artificial Neural Networks (ANNs) for predicting the refractive index of materials based on their composition. The dataset and feature engineering is done using matminer, a materials data mining library, and the machine learning model is built using TensorFlow/Keras.

Let’s go step by step through the code:

###  Importing Libraries

```python
import numpy as np
import pandas as pd
from matminer.datasets import load_dataset
from matminer.featurizers.composition import ElementProperty
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import warnings
warnings.filterwarnings('ignore')
```

- **numpy** and **pandas**: These libraries are used for handling data and numerical computations.
- **matminer**: A data mining toolkit for materials science. It provides access to datasets and tools for extracting features from material compositions.
- **scikit-learn**: This library provides tools for splitting data, scaling features, and evaluating the model using Mean Absolute Error (MAE).
- **TensorFlow Keras**: Used for building and training an artificial neural network.
- **warnings**: Warnings are suppressed to make the output cleaner.

### Loading Data

The load_dataset('dielectric_constant') function loads a materials dataset related to the dielectric constant from the matminer library.
The dataset includes structural data for various materials, which we will use to predict their refractive index (a measure of how light propagates through the material).

```python
df = load_dataset('dielectric_constant')
print("Dataset loaded with {} entries.".format(len(df)))
```
### Extracting Material Composition
```python
df['composition'] = df['structure'].apply(lambda x: x.composition)
```
The column 'structure' in the dataset contains atomic structures of the materials.
The code extracts the composition of each material using the apply function. This composition is crucial for generating features (descriptors) that will be used to train the neural network.

### Featurization using ElementProperty

```python
ep_featurizer = ElementProperty.from_preset(preset_name="magpie")
df = ep_featurizer.featurize_dataframe(df, col_id='composition', ignore_errors=True)
print("Featurization complete. Number of features: {}.".format(len(ep_featurizer.feature_labels())))
```

Featurization refers to the process of converting the material’s composition into numerical features that can be used for machine learning.
The ElementProperty featurizer from matminer generates descriptors for the chemical composition using a preset called magpie (Materials Agnostic Platform for Informatics and Exploration). These features could include atomic radii, electronegativity, and other elemental properties.
The featurize_dataframe method applies this featurizer to the dataset and appends the features as new columns in the dataframe.
The line print("Featurization complete...") confirms the process and tells us how many features (descriptors) were generated.
### Defining Features and Target Variable

```python
X = df[ep_featurizer.feature_labels()]
y = df['n']  # Refractive index
```

X: The features matrix containing the featurized composition (numerical descriptors of the material's atomic structure).
y: The target variable, which is the refractive index (denoted as 'n' in the dataset).

### Handling Missing Data and Data Splitting

```python
X = X.dropna()
y = y.loc[X.index]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Data split into training and testing sets.")
```

X.dropna(): Removes rows with missing values in the feature matrix.
The dataset is split into training (80%) and testing (20%) sets using train_test_split. The training set is used to fit the model, and the testing set is used to evaluate the model’s performance.

### Feature Scaling

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Feature scaling complete.")
```
Standardization is applied to the features using StandardScaler, which ensures that each feature has a mean of 0 and a standard deviation of 1. This is crucial for neural networks to ensure faster convergence and more stable training.
The scaler is first fit on the training data and then applied to both the training and test data.

### Building the Artificial Neural Network

```python
model = Sequential()
model.add(Dense(128, input_dim=X_train_scaled.shape[1], activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))  # Output layer for regression
print("Neural network model constructed.")
```

A Sequential model from Keras is used to define the neural network.
The input layer has 128 neurons with ReLU activation.
A second hidden layer with 64 neurons and ReLU activation is added.
The output layer consists of 1 neuron (since this is a regression task, where the network predicts a continuous value: the refractive index).
ReLU (Rectified Linear Unit) is a commonly used activation function that helps the network learn non-linear relationships between features.

### Compiling the Model

```python
model.compile(loss='mean_squared_error', optimizer='adam')
The model is compiled with the Mean Squared Error (MSE) as the loss function, which is appropriate for regression tasks.
The Adam optimizer is used, which is a variant of stochastic gradient descent that adapts the learning rate during training and generally performs well in practice.


### Training the Model

```python
history = model.fit(
    X_train_scaled, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.1,
    verbose=0  # Set to 1 to see training progress
)
print("Model training complete.")
```

The model is trained using the training data for 100 epochs with a batch size of 32. Each epoch consists of a complete pass over the training data.
Validation split: 10% of the training data is used for validation during training to monitor the model’s performance on unseen data.
verbose=0 suppresses the output, but you can set verbose=1 to see the training progress.

### Evaluating the Model

```python
y_pred = model.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
print("Model evaluation complete.")
print(f"Test Mean Absolute Error (MAE): {mae:.4f}")
```

After training, the model is evaluated on the test set.
The predictions (y_pred) are generated using the test features.
The Mean Absolute Error (MAE) between the predicted and actual refractive indices is computed as the evaluation metric. MAE is a commonly used regression metric that measures the average magnitude of the prediction errors.

### Plotting the Results

```python
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel('Actual Refractive Index')
plt.ylabel('Predicted Refractive Index')
plt.title('Actual vs Predicted Refractive Index')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.show()
```

A scatter plot is generated to compare the actual refractive index (on the x-axis) with the predicted refractive index (on the y-axis).
The diagonal dashed line represents the ideal case where the predicted values match the true values perfectly. The closer the points are to this line, the better the model’s predictions.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from sklearn.metrics import mean_squared_error

# For reproducibility
np.random.seed(42)

# Import functions to draw shapes
from skimage.draw import disk, ellipse, rectangle

### STEP 1: Generate Physically Realistic Data ###

# Simulate a simple optical tomography system (e.g., 2D grid of 32x32 pixels)
image_size = 32  # 32x32 pixel internal structure

# Let's assume we have 1000 different internal structures to simulate
n_samples = 1000

def generate_internal_structures(n_samples, image_size):
    """
    Generate physically realistic internal structures with a background medium and random inclusions.
    Inclusions are randomly placed shapes (disks, ellipses, rectangles) with different absorption coefficients.
    """
    structures = []
    for _ in range(n_samples):
        # Start with a background absorption coefficient (e.g., soft tissue)
        background_absorption = 0.01  # Low absorption coefficient
        image = np.ones((image_size, image_size)) * background_absorption

        # Randomly decide the number of inclusions (e.g., tumors or anomalies)
        n_inclusions = np.random.randint(1, 6)  # Between 1 and 5 inclusions

        for _ in range(n_inclusions):
            # Randomly choose a shape
            shape_type = np.random.choice(['disk', 'ellipse', 'rectangle'])
            # Randomly choose size and position
            if shape_type == 'disk':
                radius = np.random.randint(3, image_size // 4)
                center_x = np.random.randint(radius, image_size - radius)
                center_y = np.random.randint(radius, image_size - radius)
                rr, cc = disk((center_y, center_x), radius, shape=image.shape)
            elif shape_type == 'ellipse':
                center_x = np.random.randint(image_size // 4, 3 * image_size // 4)
                center_y = np.random.randint(image_size // 4, 3 * image_size // 4)
                major_axis = np.random.randint(5, image_size // 3)
                minor_axis = np.random.randint(3, major_axis)
                orientation = np.random.uniform(0, np.pi)
                rr, cc = ellipse(center_y, center_x, minor_axis, major_axis, shape=image.shape, rotation=orientation)
            elif shape_type == 'rectangle':
                start_x = np.random.randint(0, image_size - 5)
                start_y = np.random.randint(0, image_size - 5)
                end_x = np.random.randint(start_x + 5, min(start_x + image_size // 4, image_size))
                end_y = np.random.randint(start_y + 5, min(start_y + image_size // 4, image_size))
                rr, cc = rectangle(start=(start_y, start_x), end=(end_y, end_x), shape=image.shape)

            # Randomly choose absorption coefficient for the inclusion (e.g., higher than background)
            inclusion_absorption = np.random.uniform(0.05, 0.1)
            # Add the inclusion to the image
            image[rr, cc] = inclusion_absorption

        structures.append(image)

    return np.array(structures)

# Generate physically realistic internal structures
absorption_coefficients = generate_internal_structures(n_samples, image_size)

# Simulate the detector readings using a simple optical tomography model
# Here we use a simplified Beer-Lambert law model for light absorption

def simulate_light_propagation(internal_structure):
    """
    Simulate light propagation through the medium using Beer-Lambert Law.
    This simulates light being absorbed as it passes through the material.
    We simulate a 'detector' that collects light after passing through the medium.

    This is a simplified model assuming light passes through in straight lines and is absorbed exponentially.
    """
    # Simulate detector readings based on absorption using Beer-Lambert Law
    # Light travels from one side of the grid to the opposite side
    readings = []
    for i in range(image_size):
        # Sum over rows to simulate light passing through the material in a straight line (1D integration along y-axis)
        total_absorption = np.sum(internal_structure[i, :])
        transmission = np.exp(-total_absorption)
        readings.append(transmission)
    return np.array(readings)

# Generate detector readings for all samples
X_detector = np.array([simulate_light_propagation(absorption_coefficients[i]) for i in range(n_samples)])

# Reshape the true internal structure (flatten 2D images) for training (32x32 = 1024 features)
X_true_flatten = absorption_coefficients.reshape(n_samples, -1)

### STEP 2: Train-Test Split ###

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X_detector, X_true_flatten, test_size=0.2, random_state=42)

### STEP 3: Neural Network Architecture ###

# Build a neural network to reconstruct the 32x32 internal structure from the detector readings (32 features)
model = Sequential()

# Input is the detector readings (32 features)
model.add(Dense(128, input_dim=X_detector.shape[1], activation='relu'))

# Fully connected layers with regularization to avoid overfitting
model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01)))

# Output layer: 32x32 pixels (flattened to 1024 neurons)
model.add(Dense(image_size * image_size, activation='linear'))

# Compile the model using MSE loss and Adam optimizer
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

### STEP 4: Training the Model ###

# Train the model with the training data (detector readings -> internal structure)
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), verbose=1)

### STEP 5: Evaluation and Prediction ###

# Predict the internal structures from the test set
y_pred = model.predict(X_test)

# Reshape the predictions back to 32x32 images
y_pred_images = y_pred.reshape(-1, image_size, image_size)
y_test_images = y_test.reshape(-1, image_size, image_size)

# Calculate MSE on the test set
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (MSE) on Test Set: {mse:.6f}")

### STEP 6: Visualize the Results ###

# Plot original vs reconstructed internal structures for a random test sample
sample_index = np.random.randint(0, X_test.shape[0])

plt.figure(figsize=(12, 5))

# Plot the true internal structure
plt.subplot(1, 3, 1)
plt.imshow(y_test_images[sample_index], cmap='viridis', origin='lower')
plt.title('True Internal Structure')
plt.colorbar(fraction=0.046, pad=0.04)

# Plot the reconstructed internal structure
plt.subplot(1, 3, 2)
plt.imshow(y_pred_images[sample_index], cmap='viridis', origin='lower')
plt.title('Reconstructed Internal Structure (ANN)')
plt.colorbar(fraction=0.046, pad=0.04)

# Plot the difference between true and reconstructed structures
plt.subplot(1, 3, 3)
difference = np.abs(y_test_images[sample_index] - y_pred_images[sample_index])
plt.imshow(difference, cmap='hot', origin='lower')
plt.title('Absolute Difference')
plt.colorbar(fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()

# Plot the training and validation loss over epochs
plt.figure(figsize=(8,6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.title('Training and Validation Loss Over Epochs')
plt.legend()
plt.grid(True)
plt.show()
```

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# For reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Import functions to draw shapes
from skimage.draw import disk, ellipse, rectangle

### STEP 1: Generate Physically Realistic Data ###

# Simulate a 2D grid of 64x64 pixels for higher resolution
image_size = 64  # Increased from 32 to 64 pixels

# Let's assume we have 2000 different internal structures to simulate
n_samples = 2000  # Increased sample size for better training

def generate_internal_structures(n_samples, image_size):
    """
    Generate physically realistic internal structures with a background medium and random inclusions.
    Inclusions are randomly placed shapes (disks, ellipses, rectangles) with different absorption coefficients.
    """
    structures = []
    for _ in range(n_samples):
        # Start with a background absorption coefficient (e.g., soft tissue)
        background_absorption = 0.01  # Low absorption coefficient
        image = np.ones((image_size, image_size)) * background_absorption

        # Randomly decide the number of inclusions (e.g., tumors or anomalies)
        n_inclusions = np.random.randint(1, 6)  # Between 1 and 5 inclusions

        for _ in range(n_inclusions):
            # Randomly choose a shape
            shape_type = np.random.choice(['disk', 'ellipse', 'rectangle'])
            # Randomly choose size and position
            if shape_type == 'disk':
                radius = np.random.randint(3, image_size // 8)
                center_x = np.random.randint(radius, image_size - radius)
                center_y = np.random.randint(radius, image_size - radius)
                rr, cc = disk((center_y, center_x), radius, shape=image.shape)
            elif shape_type == 'ellipse':
                center_x = np.random.randint(image_size // 8, 7 * image_size // 8)
                center_y = np.random.randint(image_size // 8, 7 * image_size // 8)
                major_axis = np.random.randint(5, image_size // 6)
                minor_axis = np.random.randint(3, major_axis)
                orientation = np.random.uniform(0, np.pi)
                rr, cc = ellipse(center_y, center_x, minor_axis, major_axis, shape=image.shape, rotation=orientation)
            elif shape_type == 'rectangle':
                start_x = np.random.randint(0, image_size - 5)
                start_y = np.random.randint(0, image_size - 5)
                end_x = np.random.randint(start_x + 5, min(start_x + image_size // 4, image_size))
                end_y = np.random.randint(start_y + 5, min(start_y + image_size // 4, image_size))
                rr, cc = rectangle(start=(start_y, start_x), end=(end_y, end_x), shape=image.shape)

            # Randomly choose absorption coefficient for the inclusion (e.g., higher than background)
            inclusion_absorption = np.random.uniform(0.05, 0.15)
            # Add the inclusion to the image
            image[rr, cc] = inclusion_absorption

        structures.append(image)

    return np.array(structures)

# Generate physically realistic internal structures
absorption_coefficients = generate_internal_structures(n_samples, image_size)

# Normalize absorption coefficients to [0, 1]
absorption_coefficients = (absorption_coefficients - absorption_coefficients.min()) / (absorption_coefficients.max() - absorption_coefficients.min())

# Simulate the detector readings using a more realistic optical tomography model
# Using Diffusion Approximation for light propagation

def simulate_light_propagation(internal_structure):
    """
    Simulate light propagation through the medium using a simplified diffusion approximation model.
    """
    # For simplicity, we'll simulate measurements from multiple sources and detectors around the boundary
    num_sources = 16  # Number of sources placed uniformly around the boundary
    readings = []

    # Generate source positions
    positions = np.linspace(0, image_size - 1, num_sources, dtype=int)
    for source_pos in positions:
        # Simulate light propagation from this source
        # For this example, we'll sum absorption coefficients along several paths
        # In a real scenario, you would use a numerical solver for the diffusion equation
        # Here we approximate by integrating along straight lines in different directions
        # This is a simplification due to computational limitations

        # Vertical propagation (downwards)
        path = internal_structure[:, source_pos]
        transmission = np.exp(-np.sum(path))
        readings.append(transmission)

        # Horizontal propagation (rightwards)
        path = internal_structure[source_pos, :]
        transmission = np.exp(-np.sum(path))
        readings.append(transmission)

        # Diagonal propagation (down-right)
        path = np.diagonal(internal_structure, offset=source_pos - image_size // 2)
        transmission = np.exp(-np.sum(path))
        readings.append(transmission)

        # Diagonal propagation (up-right)
        path = np.diagonal(np.fliplr(internal_structure), offset=source_pos - image_size // 2)
        transmission = np.exp(-np.sum(path))
        readings.append(transmission)

    return np.array(readings)

# Generate detector readings for all samples
X_detector = np.array([simulate_light_propagation(absorption_coefficients[i]) for i in range(n_samples)])

# Normalize detector readings to [0, 1]
X_detector = (X_detector - X_detector.min()) / (X_detector.max() - X_detector.min())

# Reshape the true internal structure for training (64x64 pixels)
X_true = absorption_coefficients[..., np.newaxis]  # Add channel dimension

### STEP 2: Train-Test Split ###

# Split the data into training and testing sets (80% training, 20% testing)
X_train_det, X_test_det, y_train_img, y_test_img = train_test_split(X_detector, X_true, test_size=0.2, random_state=42)

### STEP 3: Neural Network Architecture (U-Net) ###

def unet_model(input_size=(image_size, image_size, 1)):
    inputs = Input(input_size)

    # Encoding path
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    # Bottleneck
    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(c4)

    # Decoding path
    u5 = UpSampling2D((2, 2))(c4)
    u5 = Concatenate()([u5, c3])
    c5 = Conv2D(256, (3, 3), activation='relu', padding='same')(u5)
    c5 = Conv2D(256, (3, 3), activation='relu', padding='same')(c5)

    u6 = UpSampling2D((2, 2))(c5)
    u6 = Concatenate()([u6, c2])
    c6 = Conv2D(128, (3, 3), activation='relu', padding='same')(u6)
    c6 = Conv2D(128, (3, 3), activation='relu', padding='same')(c6)

    u7 = UpSampling2D((2, 2))(c6)
    u7 = Concatenate()([u7, c1])
    c7 = Conv2D(64, (3, 3), activation='relu', padding='same')(u7)
    c7 = Conv2D(64, (3, 3), activation='relu', padding='same')(c7)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c7)

    model = Model(inputs=[inputs], outputs=[outputs])

    return model

# Build the model
model = unet_model(input_size=(image_size, image_size, 1))
model.summary()

# Compile the model using Adam optimizer and combined loss
def combined_loss(y_true, y_pred):
    mse = tf.keras.losses.MeanSquaredError()(y_true, y_pred)
    ssim = 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))
    return mse + ssim

model.compile(optimizer=Adam(learning_rate=0.0001), loss=combined_loss, metrics=['mse'])

### STEP 4: Preparing Data for Training ###

# Since our model expects image inputs, we need to reshape the detector readings to match the input size
# We'll reshape them into a 4x4 grid
detector_grid_size = int(np.sqrt(X_detector.shape[1]))
X_train_det_img = X_train_det.reshape(-1, detector_grid_size, detector_grid_size, 1)
X_test_det_img = X_test_det.reshape(-1, detector_grid_size, detector_grid_size, 1)

# Resize detector images to match the internal structure size
X_train_det_img = tf.image.resize(X_train_det_img, [image_size, image_size]).numpy()
X_test_det_img = tf.image.resize(X_test_det_img, [image_size, image_size]).numpy()

# Normalize detector images
X_train_det_img = (X_train_det_img - X_train_det_img.min()) / (X_train_det_img.max() - X_train_det_img.min())
X_test_det_img = (X_test_det_img - X_test_det_img.min()) / (X_test_det_img.max() - X_test_det_img.min())

### STEP 5: Data Augmentation ###

# Create a custom data generator to apply the same augmentations to both inputs and labels
def create_augmented_generator(X_train, y_train, batch_size, seed):
    data_gen_args = dict(rotation_range=20,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         zoom_range=0.1,
                         horizontal_flip=True,
                         vertical_flip=True,
                         fill_mode='nearest')

    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)

    # Provide the same seed and keyword arguments to the flow methods
    image_generator = image_datagen.flow(
        X_train, batch_size=batch_size, seed=seed)
    mask_generator = mask_datagen.flow(
        y_train, batch_size=batch_size, seed=seed)

    # Combine generators into one which yields image and masks
    while True:
        X_batch = next(image_generator)
        y_batch = next(mask_generator)
        yield (X_batch, y_batch)

# Create the generator
batch_size = 16
seed = 42
train_generator = create_augmented_generator(X_train_det_img, y_train_img, batch_size, seed)

### STEP 6: Training the Model ###

# Define callbacks
callbacks = [
    EarlyStopping(patience=10, verbose=1, restore_best_weights=True),
    ModelCheckpoint('unet_model.keras', verbose=1, save_best_only=True)
]

# Calculate steps per epoch
steps_per_epoch = len(X_train_det_img) // batch_size

# Train the model
history = model.fit(train_generator,
                    steps_per_epoch=steps_per_epoch,
                    epochs=100,
                    validation_data=(X_test_det_img, y_test_img),
                    callbacks=callbacks,
                    verbose=1)

### STEP 7: Evaluation and Prediction ###

# Predict the internal structures from the test set
y_pred_img = model.predict(X_test_det_img)

# Ensure the predictions are within [0, 1]
y_pred_img = np.clip(y_pred_img, 0, 1)

# Calculate evaluation metrics
mse = mean_squared_error(y_test_img.flatten(), y_pred_img.flatten())
psnr = peak_signal_noise_ratio(y_test_img, y_pred_img, data_range=1)
ssim = structural_similarity(y_test_img.squeeze(), y_pred_img.squeeze(), multichannel=False)

print(f"Mean Squared Error (MSE) on Test Set: {mse:.6f}")
print(f"Peak Signal-to-Noise Ratio (PSNR) on Test Set: {psnr:.2f} dB")
print(f"Structural Similarity Index (SSIM) on Test Set: {ssim:.4f}")

### STEP 8: Visualize the Results ###

# Plot original vs reconstructed internal structures for a random test sample
sample_index = np.random.randint(0, X_test_det_img.shape[0])

plt.figure(figsize=(15, 5))

# Plot the true internal structure
plt.subplot(1, 4, 1)
plt.imshow(y_test_img[sample_index, :, :, 0], cmap='viridis', origin='lower')
plt.title('True Internal Structure')
plt.axis('off')
plt.colorbar(fraction=0.046, pad=0.04)

# Plot the detector readings (reshaped)
plt.subplot(1, 4, 2)
plt.imshow(X_test_det_img[sample_index, :, :, 0], cmap='gray', origin='lower')
plt.title('Detector Readings')
plt.axis('off')
plt.colorbar(fraction=0.046, pad=0.04)

# Plot the reconstructed internal structure
plt.subplot(1, 4, 3)
plt.imshow(y_pred_img[sample_index, :, :, 0], cmap='viridis', origin='lower')
plt.title('Reconstructed Internal Structure')
plt.axis('off')
plt.colorbar(fraction=0.046, pad=0.04)

# Plot the absolute difference
plt.subplot(1, 4, 4)
difference = np.abs(y_test_img[sample_index, :, :, 0] - y_pred_img[sample_index, :, :, 0])
plt.imshow(difference, cmap='hot', origin='lower')
plt.title('Absolute Difference')
plt.axis('off')
plt.colorbar(fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()

# Plot the training and validation loss over epochs
plt.figure(figsize=(8,6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Combined Loss (MSE + 1 - SSIM)')
plt.title('Training and Validation Loss Over Epochs')
plt.legend()
plt.grid(True)
plt.show()
```

