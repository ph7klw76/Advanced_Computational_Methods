# Grover Adaptive Search in Quantum Computing


# Introduction

Grover’s algorithm is one of the most well-known quantum algorithms, famous for its ability to perform unstructured search tasks quadratically faster than any classical algorithm. However, Grover Adaptive Search (GAS) extends this capability to a broader range of optimization and search tasks by making Grover’s algorithm more adaptable and efficient for real-world problems. GAS is particularly valuable for optimization problems where the solution space is vast and complex, and the desired solution is not simply "any match" but the optimal match based on a given objective function.

This article delves into Grover Adaptive Search in quantum computing, detailing the principles behind it, the mechanisms of how it works, and the specific mathematical underpinnings. By leveraging an adaptive version of Grover’s algorithm, Grover Adaptive Search is well-suited for scenarios where flexibility and iterative refinement in searching are key.

## Background: Grover’s Algorithm and Quadratic Speedup

Before diving into Grover Adaptive Search, it's essential to understand the basics of Grover's algorithm. Grover’s algorithm provides a quantum advantage in unstructured search problems, where the goal is to find a specific item in an unsorted database. Classical algorithms solve this problem with $O(N)$ queries (where $N$ is the number of items), while Grover’s algorithm requires only $O(\sqrt{N})$ queries, achieving a quadratic speedup.

### Grover's Algorithm Mechanics

- **State Initialization**: Start with an equal superposition over all $N$ possible states.
- **Oracle Application**: Use a quantum oracle to mark the solution state by flipping its phase. The oracle is designed to identify states that satisfy a specific condition, typically encoded in a Boolean function.
- **Amplitude Amplification**: Repeatedly apply a process called the Grover operator, which amplifies the probability amplitude of the marked solution state. This operator consists of a reflection about the mean, allowing the amplitude of the target state to grow with each iteration.
- **Measurement**: After approximately $\sqrt{N}$ iterations, measuring the system will yield the marked state with high probability.

Grover’s algorithm is optimal for finding a single solution in an unsorted set. However, in many real-world applications, the goal is to find the best solution (according to an objective function), not simply any solution that satisfies a condition. This is where Grover Adaptive Search comes in.

## How Does a Quantum Oracle Work?

A quantum oracle works by flipping the phase or modifying specific quantum states based on whether they satisfy a condition. Generally, a quantum oracle is represented as a unitary operator $O_f$ that takes as input a quantum state $|x\rangle$ and transforms it according to a given Boolean function $f(x)$:

$$
O_f |x\rangle = (-1)^{f(x)} |x\rangle
$$

In this form:

- **If $f(x) = 0$**: The oracle leaves the state $|x\rangle$ unchanged.
- **If $f(x) = 1$**: The oracle flips the phase of $|x\rangle$ (multiplies it by $-1$).

This phase flip effectively "marks" the solution state(s) by giving them a unique sign, which can then be detected through quantum interference or amplitude amplification techniques, as seen in Grover’s algorithm.

## The Need for Grover Adaptive Search

In optimization problems, we are often interested in finding a minimum or maximum value of a function rather than just finding any state that satisfies a condition. Traditional Grover’s algorithm is not well-suited for optimization because it only finds solutions in a single round. In contrast, Grover Adaptive Search introduces an iterative, adaptive approach that narrows down the search space in each step based on previous results, making it highly effective for finding optimal solutions.

GAS iteratively applies Grover’s search to look for solutions that satisfy increasingly strict conditions on the objective function, thus honing in on the optimal solution over multiple rounds.

## Grover Adaptive Search Mechanism

1. **Define an Objective Function**: First, define a quantum oracle that encodes an objective function $f(x)$. The function $f(x)$ assigns a “score” or “value” to each possible solution $x$.

2. **Initial Bound Setting**: Establish an initial bound or threshold $T$ on the objective function, beyond which the solution is considered desirable. This bound is adjusted adaptively throughout the search process.

3. **Iterative Refinement**:

   - **Grover Iteration with Updated Oracle**: In each iteration, GAS uses Grover’s algorithm to search for states $x$ that satisfy the condition $f(x) \geq T$ (or $f(x) \leq T$ for minimization problems).
   - **Update Bound**: Once a solution is found, the bound $T$ is updated to be closer to the value of the current solution, narrowing down the search.
   - **Repeat**: This process repeats, tightening $T$ iteratively until the best (or optimal) solution is identified.

4. **Termination**: The search terminates once further iterations do not yield an improvement in the objective function value or when $T$ converges to the optimal value. At this point, the system is measured to retrieve the optimal solution with high probability.

## Mathematical Framework of Grover Adaptive Search

Let’s examine the mathematical details of each step in GAS, focusing on the use of Grover's operator with an adaptive threshold.

### Step 1: Objective Function Oracle

For a given optimization problem, we define an objective function $f(x)$, where $x$ is a candidate solution represented as a quantum state. The objective function maps each $x$ to a value $f(x)$, with the goal of maximizing or minimizing $f(x)$.

To implement GAS, we design an oracle $O_T$ that marks states based on whether they meet a threshold $T$ in the objective function:

$$
O_T |x\rangle =
\begin{cases} 
      -|x\rangle, & \text{if } f(x) \geq T, \\
      |x\rangle, & \text{otherwise.}
   \end{cases}
$$

The oracle flips the phase of states where $f(x) \geq T$, marking them as potential solutions.

### Step 2: Grover Iteration and Amplitude Amplification

Using the oracle $O_T$, we apply Grover's operator to amplify the amplitude of states that satisfy $f(x) \geq T$. The Grover operator $G = -HO_T H$ is applied iteratively, where $H$ represents the Hadamard transformation that performs a reflection around the mean.

After $O(\sqrt{N})$ iterations of the Grover operator, the marked states’ amplitude is amplified, making it more likely that measurement will yield one of these states.

### Step 3: Updating the Bound $T$

Once a solution satisfying $f(x) \geq T$ is found, we update $T$ to be closer to $f(x)$, refining the bound. For example, if we are maximizing, we set $T = f(x)$ where $f(x)$ is the best solution found so far.

This adaptive update ensures that in each subsequent round of Grover search, we are looking for solutions that are increasingly close to the optimal value, honing in on the best solution through successive refinements of $T$.

## Example of Grover Adaptive Search for Optimization

Consider an optimization problem where we aim to maximize a function $f(x)$ over a domain of possible states represented by $x$. For simplicity, let $f(x)$ be defined over four states: $|00\rangle$, $|01\rangle$, $|10\rangle$, and $|11\rangle$, with corresponding values:

- $f(|00\rangle) = 1$
- $f(|01\rangle) = 3$
- $f(|10\rangle) = 7$
- $f(|11\rangle) = 5$

### Initial Bound

- **Threshold**: Set an initial threshold $T=0$, meaning any state can be a candidate.

### First Grover Search

- **Objective**: Apply Grover's algorithm to amplify states with $f(x) \geq T = 0$.
- **Result**: This search may yield $|10\rangle$ as a solution since it has the highest value (7).
- **Update Threshold**: Set $T = 7$.

### Second Grover Search with Updated Threshold

- **Updated Threshold**: Now set $T = 7$.
- **Objective**: Apply Grover’s search again, looking for states satisfying $f(x) \geq 7$.
- **Result**: If the state $|10\rangle$ is found again, it confirms that $f(|10\rangle) = 7$ is the maximum.

This iterative process narrows the search space until we confirm that $|10\rangle$ is the optimal solution.

## Advantages of Grover Adaptive Search

- **Flexible for Optimization**: Unlike standard Grover's algorithm, GAS is designed for optimization, making it suitable for problems where the goal is to find the best possible solution rather than any solution.
- **Adaptability**: The algorithm's adaptive update mechanism allows it to zero in on optimal solutions without needing to know the exact location of the solution state in advance.
- **Efficient Iteration**: By iteratively updating the threshold $T$ and using amplitude amplification, GAS focuses computational resources on progressively smaller regions of the solution space, enhancing efficiency.

## Applications of Grover Adaptive Search

Grover Adaptive Search has potential applications in several domains, particularly where optimization is essential:

- **Resource Allocation**: Finding the optimal allocation of resources across various tasks to maximize output.
- **Portfolio Optimization**: Selecting the best mix of assets for financial portfolios to maximize return while minimizing risk.
- **Scheduling Problems**: Determining the optimal scheduling of jobs, projects, or operations to optimize time and resources.
- **Supply Chain Management**: Optimizing logistics and supply chain networks to reduce costs and improve efficiency.

# Grover Adaptive Search (GAS) in Applied Physics

Grover Adaptive Search (GAS) is leveraged in applied physics to tackle complex optimization problems, especially those requiring optimal configurations, minimizing energy states, or identifying specific solutions in large datasets. Many problems in applied physics can be framed as finding an optimal solution within a large set of possibilities, where a traditional search would be prohibitively slow. GAS provides a more efficient quantum approach by iteratively refining the search criteria based on feedback from previous results, allowing for faster convergence on optimal or near-optimal solutions.

### Applications of GAS in Applied Physics

1. **Molecular and Materials Simulation**

   - **Problem**: Finding the ground-state configuration of molecules, where the system's total energy is minimized, is critical for understanding molecular stability and designing materials with desired properties.
   - **GAS Application**: By iteratively refining search bounds, GAS focuses on configurations with progressively lower energies. For example, when simulating a new material, GAS can help identify the atomic arrangement that yields the lowest energy, enabling predictions about stability, conductivity, or other properties.

2. **Optimization of Physical Systems in Engineering**

   - **Problem**: Engineering challenges often require finding optimal configurations to maximize efficiency or minimize waste, such as designing aerodynamic shapes, optimizing power distribution networks, or configuring cooling systems.
   - **GAS Application**: GAS identifies optimal solutions by adjusting search parameters to narrow possibilities for the most efficient design, such as the optimal shape for an aircraft wing or the minimal-energy layout of a cooling network. This approach speeds up design iterations and enhances system efficiency.

3. **Magnetic Material Configuration and Spin Glasses**

   - **Problem**: In condensed matter physics, finding the lowest-energy configuration of spins in disordered magnetic systems, such as spin glasses, is crucial for understanding magnetic properties.
   - **GAS Application**: GAS adaptively narrows down configurations to locate those with lower energy, facilitating the search for ground states. This adaptability helps avoid local minima, providing insights into the behavior of magnetic materials under varying conditions, which is valuable in quantum magnetism, superconductivity, and quantum computing materials.

4. **Nuclear Physics and Particle Collision Optimization**

   - **Problem**: Optimizing particle interaction pathways or collision parameters is essential for studying fundamental particle behavior or simulating nuclear reactor conditions.
   - **GAS Application**: By iteratively refining search conditions, GAS identifies the most probable or efficient collision outcomes based on energy conservation and reaction thresholds. In particle accelerators, GAS helps in selecting configurations that maximize the probability of observing specific high-energy collisions, aiding in the discovery of new particles or interactions.

5. **Quantum State Preparation and Quantum Control**

   - **Problem**: Precise control over quantum states is vital for achieving specific states, such as entangled or ground states, which are foundational in quantum computing and quantum physics.
   - **GAS Application**: GAS iteratively refines control parameters, such as magnetic fields or laser pulses, to increase the likelihood of achieving the desired state. This is essential in quantum computing for precise state preparation, ensuring robust qubit performance and facilitating the execution of quantum algorithms.


```python
import pennylane as qml
from pennylane import numpy as np

# Define the number of qubits representing the problem
n_qubits = 4  # Number of qubits representing the configurations
n_iterations = 1  # Number of Grover iterations per adaptive step

# Define the energies of the configurations
# For simplicity, we use the Hamming weight (number of ones) as the energy
def energy(bitstring):
    return sum(bitstring)

# Generate all possible configurations and their energies
from itertools import product
configurations = list(product([0, 1], repeat=n_qubits))
energies = [energy(config) for config in configurations]

# Find the maximum energy (initial threshold)
max_energy = max(energies)

# Number of bits needed to represent energies
n_energy_bits = int(np.ceil(np.log2(max_energy + 1)))

# Total number of qubits (configuration qubits + energy qubits + one ancilla qubit)
total_qubits = n_qubits + n_energy_bits + 1

# Update the device
dev = qml.device('default.qubit', wires=total_qubits, shots=1)

# Update the oracle function to avoid deprecation warnings
def oracle(threshold):
    def oracle_circuit():
        # Compute the energy and store it in the energy register
        for i in range(n_energy_bits):
            qml.PauliX(wires=n_qubits + i)  # Initialize energy qubits to |1>

        # Simulate energy computation (this is a placeholder for the actual energy computation circuit)
        for i in range(n_qubits):
            qml.CNOT(wires=[i, n_qubits + i % n_energy_bits])

        # Subtract threshold (this is a simplified representation)
        binary_threshold = np.binary_repr(threshold, width=n_energy_bits)
        for i, bit in enumerate(binary_threshold[::-1]):
            if bit == '1':
                qml.PauliX(wires=n_qubits + i)

        # Apply multi-controlled Toffoli gate to flip the ancilla qubit if energy < threshold
        qml.MultiControlledX(
            wires=(*range(n_qubits, n_qubits + n_energy_bits), total_qubits - 1),
            control_values=[0] * n_energy_bits
        )

        # Add threshold back (uncompute)
        for i, bit in enumerate(binary_threshold[::-1]):
            if bit == '1':
                qml.PauliX(wires=n_qubits + i)

        # Uncompute the energy computation
        for i in reversed(range(n_qubits)):
            qml.CNOT(wires=[i, n_qubits + i % n_energy_bits])

        for i in range(n_energy_bits):
            qml.PauliX(wires=n_qubits + i)  # Reset energy qubits to |0>
    return oracle_circuit

# Update the diffusion operator to avoid deprecation warnings
def diffusion_operator():
    # Apply Hadamard gates to configuration qubits
    for i in range(n_qubits):
        qml.Hadamard(wires=i)
    # Apply Pauli-X gates to configuration qubits
    for i in range(n_qubits):
        qml.PauliX(wires=i)
    # Apply multi-controlled Z gate
    qml.MultiControlledX(
        wires=(*range(n_qubits), total_qubits - 1),
        control_values=[1] * n_qubits
    )
    # Apply Pauli-X gates to configuration qubits
    for i in range(n_qubits):
        qml.PauliX(wires=i)
    # Apply Hadamard gates to configuration qubits
    for i in range(n_qubits):
        qml.Hadamard(wires=i)

# Update the part where the sample is converted to a list
def grover_adaptive_search():
    threshold = max_energy + 1
    best_solution = None
    best_energy = threshold
    while threshold > 0:
        threshold -= 1
        print(f"\nCurrent threshold: {threshold}")
        # Check if there are solutions below the threshold
        num_solutions = sum(1 for e in energies if e < threshold)
        if num_solutions == 0:
            continue
        num_iterations = int(np.floor((np.pi / 4) * np.sqrt(2 ** n_qubits / num_solutions)))
        num_iterations = max(1, num_iterations)
        print(f"Number of Grover iterations: {num_iterations}")

        @qml.qnode(dev)
        def circuit():
            # Initialize all qubits
            for i in range(n_qubits):
                qml.Hadamard(wires=i)
            # Initialize ancilla qubit to |-> state
            qml.Hadamard(wires=total_qubits - 1)
            qml.PauliX(wires=total_qubits - 1)

            # Apply Grover iterations
            for _ in range(num_iterations):
                # Oracle
                oracle_circuit = oracle(threshold)
                oracle_circuit()
                # Diffusion operator
                diffusion_operator()

            # Measure the configuration qubits
            return qml.sample(wires=range(n_qubits))

        # Run the circuit
        sample = circuit()
        bitstring = sample.tolist()  # Convert NumPy array to list
        sample_energy = energy(bitstring)
        print(f"Sampled configuration: {bitstring}, Energy: {sample_energy}")
        if sample_energy < best_energy:
            best_energy = sample_energy
            best_solution = bitstring
            print(f"Found new best solution with energy {best_energy}: {best_solution}")
        else:
            print("No better solution found.")
            break
    print(f"\nOptimal solution found: {best_solution} with energy {best_energy}")

# Run the Grover Adaptive Search
grover_adaptive_search()
```

Detail explanation as below

### Oracle Function
The oracle function is a crucial part of Grover's algorithm. It is responsible for marking the states that meet a certain condition—in this case, states with energy below a given threshold.

1. **Initialization of Energy Qubits**: The energy qubits are initialized to the state |1>. This is done to prepare them for the subsequent operations.
2. **Energy Computation Simulation**: This step simulates the computation of the energy of the current state. In a real implementation, this would involve a series of quantum gates that calculate the energy based on the problem's specifics.
3. **Threshold Subtraction**: The binary representation of the threshold is used to apply Pauli-X gates to the corresponding qubits. This effectively subtracts the threshold from the computed energy.
4. **Multi-Controlled Toffoli Gate**: A multi-controlled Toffoli gate (also known as a multi-controlled X gate) is applied. This gate flips an ancilla qubit if the energy is below the threshold, marking the state.
5. **Uncomputation**: The threshold is added back, and the energy computation is uncomputed to reset the energy qubits to their initial state. This ensures that the qubits are ready for the next iteration.

### Diffusion Operator
The diffusion operator is used to amplify the probability of the marked states, making them more likely to be measured.

1. **Hadamard Gates**: Hadamard gates are applied to all configuration qubits to create a superposition of all possible states.
2. **Pauli-X Gates**: Pauli-X gates are applied to invert the states of the qubits.
3. **Multi-Controlled Z Gate**: A multi-controlled Z gate is applied to invert the phase of the |111...1> state. This is a crucial step in the diffusion process.
4. **Pauli-X Gates**: Pauli-X gates are applied again to revert the inversion done earlier.
5. **Hadamard Gates**: Hadamard gates are applied again to complete the diffusion operator. This step ensures that the marked states are amplified.

### Grover Adaptive Search
The Grover Adaptive Search algorithm iteratively reduces the threshold and applies Grover's algorithm to find the optimal solution.

1. **Initialize Threshold**: The threshold is initialized to a value higher than the maximum possible energy. This ensures that all states are initially considered.
2. **Iterate Over Thresholds**: The threshold is decremented in each iteration, progressively narrowing down the search space.
3. **Check for Solutions**: The number of solutions below the current threshold is counted. If no solutions are found, the threshold is decremented further.
4. **Calculate Grover Iterations**: The number of Grover iterations is calculated based on the number of solutions. This is determined using the formula involving the square root of the ratio of the total number of states to the number of solutions.
5. **Define Quantum Circuit**: A quantum circuit is defined to implement the Grover iterations.
6. **Initialize Qubits**: All qubits are initialized to the superposition state using Hadamard gates.
7. **Apply Grover Iterations**: The oracle and diffusion operator are applied for the calculated number of iterations. This amplifies the probability of the marked states.
8. **Measure Qubits**: The configuration qubits are measured to obtain a sample. This sample represents a potential solution.
9. **Evaluate Sample**: The energy of the sampled configuration is evaluated. If the energy is lower than the best energy found so far, the best solution and energy are updated.
10. **Print Results**: The optimal solution and its energy are printed at the end of the search.

### Summary
- **Oracle Function**: Marks states with energy below a threshold using multi-controlled Toffoli gates.
- **Diffusion Operator**: Amplifies the probability of marked states using a series of quantum gates.
- **Grover Adaptive Search**: Iteratively reduces the threshold and applies Grover's algorithm to find the optimal solution. It initializes qubits, applies Grover iterations, and measures the configuration qubits to find the best solution.


