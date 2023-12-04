# Parity Classification

## Information

- Author: Sjoerd Vink
- Studentnumber: 2427021
- Subject: Capita selecta on quantum middleware
- Supervisor: Nishant Saurabh
- Date: December 2023

## Explanation

This workflow \cite{parity} utilizes a variational quantum classifier to learn the parity function, leveraging the capabilities of Pennylane and Covalent. The core of the process involves the construction of a variational quantum circuit, comprising rotation and CNOT gates in a layered architecture, inspired by seminal works in quantum computing. Data encoding into the quantum circuit is achieved through Pennylane's BasisState function. The workflow defines a quantum node that integrates state preparation with the layered circuit structure, processing the input data and trainable parameters. A variational classifier, extending the quantum circuit's output with a bias term, is employed alongside a mean square loss function as the cost metric. The training employs a NesterovMomentumOptimizer, optimizing the model through iterative weight and bias updates over multiple epochs and batch processing. The workflow culminates in the model's performance evaluation, emphasizing accuracy improvements throughout the training iterations. This approach showcases the integration of quantum computing techniques in machine learning.

## Run the workflow

- python3.8 -m venv .venv
- source .venv/bin/activate
- pip install -r requirements.txt
- covalent start
- python script.py
