<h1>VQC workflow</h1>

Author: Josh Bleijenberg
Studentnumber: 6803371
Subject: Capita selecta on quantum middleware
Supervisor: Nishant Saurabh
Date: December 2023

<h2>Explanation</h2>
This workflow uses a Variational Quantum Classifier (VQC) to classify data from the Iris dataset. It involves preparing the data, defining quantum circuits (ansatz), and using the VQC algorithm for training and testing. The quantum aspect of this workflow involves using quantum circuits and a Variational Quantum Classifier (VQC) for processing and classifying data. Quantum circuits are used for feature mapping and constructing the ansatz, which are essential components of the VQC. On the other hand, the classical part includes data preprocessing using standard machine learning techniques, like MinMax scaling and train-test splitting. The workflow is designed to experiment with different configurations, varying the number of qubits, and the iterations for optimization. The results of these experiments, including the evolution of the cost function over iterations and computational performance metrics (time, CPU usage, memory usage), are systematically collected and saved to a CSV file. This structured approach facilitates a detailed evaluation of the QAOA's effectiveness under different circumstances.


<h2>Run the workflow</h2>

1. Prepare local QuantumServerless infrastructure
    i. Install Docker If Docker is not installed on your system, follow the directions on the Docker website to install Docker on your system.
    ii. Clone the Quantum Serverless repository: git clone https://github.com/Qiskit-Extensions/quantum-serverless.git
    iii. Run QuantumServerless infrastructure Execute Docker Compose using the following commands. (Note: Make sure to stop any running Jupyter Notebook servers before proceeding.)
    - cd quantum-serverless/
    - sudo docker compose --profile jupyter up
2. Python3.11 -m venv .venv
3. source .venv/bin/activate
4. pip install -r requirements.txt
5. Run the runner.ipynb


    
    
