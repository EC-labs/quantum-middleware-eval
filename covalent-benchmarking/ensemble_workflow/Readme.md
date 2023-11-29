# Ensemble workflow

## Information

- Author: Sjoerd Vink
- Studentnumber: 2427021
- Subject: Capita selecta on quantum middleware
- Supervisor: Nishant Saurabh
- Date: December 2023

## Explanation

This workflow presents a sophisticated hybrid quantum-classical machine learning approach for binary image classification, particularly distinguishing between images of cats and dogs. At its core, the workflow integrates a custom quantum circuit within a neural network, specifically modifying a ResNet model by replacing its final layer with a quantum layer. This layer leverages quantum computing principles to compute expectation values based on input parameters. Key features of the workflow include data preparation from Kaggle's 'Dogs vs. Cats' dataset, a custom backward pass applying the parameter shift rule for quantum gradients, and standard optimization techniques such as the Adam optimizer and loss functions. The process encompasses not only the training of the neural network but also its evaluation on a test dataset, assessing performance metrics like accuracy and loss. The workflow is designed for flexibility, allowing for experimentation with various configurations like the number of qubits, epochs, and batch sizes. The culmination of the process involves recording the outcomes, including training metrics and prediction results, into a CSV file, facilitating comprehensive analysis and review. This hybrid approach exemplifies the integration of quantum computing elements into classical deep learning architectures, showcasing the potential for advanced computational techniques in image classification tasks.

## Run the workflow

- python3.8 -m venv .venv
- source .venv/bin/activate
- pip install -r requirements.txt
- covalent start
- python script.py
