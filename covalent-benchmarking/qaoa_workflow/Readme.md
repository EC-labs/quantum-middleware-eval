# QAOA to Solve the Max-Cut Problem

## Information

- Author: Sjoerd Vink
- Studentnumber: 2427021
- Subject: Capita selecta on quantum middleware
- Supervisor: Nishant Saurabh
- Date: December 2023

## Explanation

This workflow implements a quantum machine learning approach to solve the Max-Cut problem using the Quantum Approximate Optimization Algorithm (QAOA). The process starts by generating random graphs, each specified by a certain number of qubits and a probability parameter, which define the connectivity between nodes. For each graph, it constructs a QAOA circuit comprising cost and mixer Hamiltonians. The QAOA circuit, defined using Pennylane, incorporates parametric quantum layers that apply cost and mixer operators at varying depths. A key feature of this workflow is the optimization of QAOA parameters to minimize the expected value of the cost Hamiltonian, which corresponds to finding the optimal solution to the Max-Cut problem for the given graph. It employs various quantum-aware optimization algorithms, including gradient descent, Adagrad, Momentum, and Adam, to iteratively update the parameters and reduce the cost function's value. The workflow is designed to experiment with different configurations, varying the number of qubits, the iterations for optimization, and the parameter initialization range. It is capable of dispatching multiple instances of the problem with varying conditions, allowing for a comprehensive analysis of the QAOA's performance across different scenarios. The results of these experiments, including the evolution of the cost function over iterations and computational performance metrics (time, CPU usage, memory usage), are systematically collected and saved to a CSV file. This structured approach facilitates a detailed evaluation of the QAOA's effectiveness in solving the Max-Cut problem under different graph conditions and optimization strategies, providing valuable insights for quantum computing applications in combinatorial optimization.

## Run the workflow

- python3.8 -m venv .venv
- source .venv/bin/activate
- pip install -r requirements.txt
- covalent start
- python script.py
