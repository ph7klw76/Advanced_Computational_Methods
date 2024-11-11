# Computational Thinking
## Introduction
Computational Thinking (CT) is an approach to problem-solving rooted in computer science, focusing on breaking down complex problems, identifying patterns, abstracting key elements, and formulating step-by-step solutions that can be executed by a computer. This framework can be applied to many domains, ranging from science and engineering to social sciences and data analysis. In this blog, we provide a detailed mathematical exploration of CT, demonstrating its power through concrete examples.

### 1. Key Components of Computational Thinking
1.1 Decomposition
Definition: Decomposition is the process of breaking a complex problem into smaller, more manageable parts, making it easier to analyze and solve.

Example: Solving a Linear Algebra Problem
Problem Statement: Solve the linear system:
$Ax = b$,
where $A \in R^{n \times n}$ is a matrix, $x \in R^n$ is the vector of unknowns, and $b \in R^n$ is a known vector.

Decomposition Steps:

Matrix Factorization: If possible, decompose $A$ into simpler matrices, such as an LU decomposition:

$$
A = LU
$$

where $L$ is a lower triangular matrix and $U$ is an upper triangular matrix.
Solve Subproblems:
First, solve the system $Ly = b$ using forward substitution.
Then, solve $Ux = y$ using backward substitution.
Mathematical Derivation: Consider a system where:

$$
A = \begin{bmatrix} 2 & 3 \\
1 & 4 \end{bmatrix}, \quad b = \begin{bmatrix} 5 \\
6 \end{bmatrix}.
$$

We perform LU decomposition:

$$
L = \begin{bmatrix} 1 & 0 \\
0.5 & 1 \end{bmatrix}, \quad U = \begin{bmatrix} 2 & 3 \\
0 & 2.5 \end{bmatrix}.
$$

To solve $Ly = b$:

$$
\begin{bmatrix} 1 & 0 \\
0.5 & 1 \end{bmatrix} \begin{bmatrix} y_1 \\
y_2 \end{bmatrix} = \begin{bmatrix} 5 \\
6 \end{bmatrix}.
$$

Forward substitution gives:

$$
y_1 = 5, \quad y_2 = 6 - 0.5 \times 5 = 3.5.
$$

Next, solve $Ux = y$:

$$
\begin{bmatrix} 2 & 3 \\
0 & 2.5 \end{bmatrix} \begin{bmatrix} x_1 \\
x_2 \end{bmatrix} = \begin{bmatrix} 5 \\
3.5 \end{bmatrix}.
$$

Backward substitution yields:

$$
x_2 = \frac{3.5}{2.5} = 1.4, \quad x_1 = \frac{5 - 3 \times 1.4}{2} = 0.4.
$$

Thus, the solution is:

$$
x = \begin{bmatrix} 0.4 \\
1.4 \end{bmatrix}.
$$

### 1.2 Abstraction
Definition: Abstraction focuses on reducing complexity by filtering out unnecessary details and focusing on the essential aspects of a problem.

Example: Graph Representation of a Network
Problem Statement: Consider a social network where users are connected to each other. The goal is to analyze connectivity between users.

Abstraction:
Represent the network as a graph $G = (V, E)$, where $V$ is a set of vertices (nodes) representing users, and $E$ is a set of edges representing connections.
Ignore irrelevant details such as user attributes (age, location) and focus on connectivity.

Mathematical Representation:
Suppose we have four users and connections represented as edges:

User 1 is connected to User 2.
User 2 is connected to User 3.
User 3 is connected to User 4.
Represent the network using an adjacency matrix $A$:

$$
A = \begin{bmatrix} 0 & 1 & 0 & 0 \\
1 & 0 & 1 & 0 \\
0 & 1 & 0 & 1 \\
0 & 0 & 1 & 0 \end{bmatrix}.
$$

Here, $A_{ij} = 1$ indicates a connection between users $i$ and $j$.

Application:
Use graph algorithms such as Breadth-First Search (BFS) to determine connectivity or find the shortest path between users.

### 1.3 Pattern Recognition
Definition: Pattern recognition involves identifying similarities or trends within data to simplify complex problems and leverage existing solutions.

Example: Identifying Sequences in Data
Problem Statement: Given the sequence of numbers:
$1, 4, 9, 16, 25, \ldots$
identify the pattern and provide a formula for the $n$-th term.

Solution:
Recognize that the sequence represents perfect squares:
$1^2, 2^2, 3^2, 4^2, 5^2, \ldots$
The formula for the $n$-th term is:
$a_n = n^2.$

Mathematical Verification:
For $n = 6$:
$a_6 = 6^2 = 36.$

### 1.4 Algorithmic Thinking
Definition: Algorithmic thinking involves developing step-by-step procedures or algorithms to solve problems.

Example: Sorting a List
Problem Statement: Sort the list $[5, 3, 8, 4, 2]$ using the Bubble Sort algorithm.

Algorithm:

Compare adjacent elements in the list.
Swap them if they are in the wrong order.
Repeat the process until the list is sorted.
Mathematical Explanation:
Pass 1:

Compare 5 and 3; swap. List becomes $[3, 5, 8, 4, 2]$.
Compare 5 and 8; no swap.
Compare 8 and 4; swap. List becomes $[3, 5, 4, 8, 2]$.
Compare 8 and 2; swap. List becomes $[3, 5, 4, 2, 8]$.
Pass 2:

Compare 3 and 5; no swap.
Compare 5 and 4; swap. List becomes $[3, 4, 5, 2, 8]$.
Compare 5 and 2; swap. List becomes $[3, 4, 2, 5, 8]$.
Continue until sorted list $[2, 3, 4, 5, 8]$ is obtained.

## 2. Applying Computational Thinking to Real-World Problems
2.1 Case Study: Traveling Salesman Problem (TSP)
Problem Statement: Given a list of cities and the distances between them, find the shortest route that visits each city exactly once and returns to the origin city.

### Application of Computational Thinking:

Decomposition: Divide the problem into smaller subproblems, such as finding the shortest path between two cities.
Abstraction: Represent cities as nodes and distances as edges in a graph.
Pattern Recognition: Recognize known substructures, such as Hamiltonian cycles.
Algorithmic Thinking: Use dynamic programming to find the optimal solution.
Mathematical Derivation:
Let $C(S, i)$ represent the minimum cost to visit all nodes in set $S$ ending at node $i$.
Recursive relation:
$C(S, i) = \min_{j \in S, j \neq i} [C(S \setminus {i}, j) + d(j, i)],$
where $d(j, i)$ is the distance between nodes $j$ and $i$.

### 2.2 Case Study: Image Compression Using Singular Value Decomposition (SVD)
Problem Statement: Compress an image while retaining essential features.

Application of Computational Thinking:

Decomposition: Separate the image matrix $A$ into components using SVD.
Abstraction: Focus on the largest singular values to retain most information.
Algorithmic Thinking: Compute the SVD of $A$:
$A = U \Sigma V^T,$
where $U$ and $V$ are orthogonal matrices, and $\Sigma$ is a diagonal matrix of singular values.
Mathematical Explanation:
Keep only the top $k$ singular values:
$A_k = U_k \Sigma_k V_k^T.$
This reduces storage and computation, approximating the original image with minimal loss of information.

## 3. Advantages and Challenges of Computational Thinking
Advantages:

Scalability: Decomposes complex problems into manageable parts.
Reusability: Leverages known solutions for similar problems.
Efficiency: Encourages optimal algorithm design.
Challenges:

Complexity of Abstraction: Balancing simplicity and accuracy.
Scalability Issues: Not all problems decompose easily.
Bias in Pattern Recognition: Incorrect patterns lead to suboptimal solutions.
