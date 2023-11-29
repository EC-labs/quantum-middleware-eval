# Iris workflow

## Information

- Author: Sjoerd Vink
- Studentnumber: 2427021
- Subject: Capita selecta on quantum middleware
- Supervisor: Nishant Saurabh
- Date: December 2023

## Explanation

This code represents a comprehensive quantum machine learning workflow for classification, specifically applied to the Iris dataset. It utilizes a quantum neural network (QNN) model, built and optimized using Pennylane, a quantum computing library. The primary task of this workflow is to classify data points from the Iris dataset into distinct classes based on their features. The workflow begins by processing the dataset to extract features and labels, followed by a training-validation split. A quantum circuit is defined for the QNN, which includes state preparation and layer functions to construct the QNN architecture. The main components of the QNN involve applying rotation gates and CNOT gates to encode data into a quantum state and process it through the network. The output of the circuit is used in a variational classifier, which is optimized using a custom loss function and a Nesterov Momentum Optimizer. The training process involves iteratively updating the weights of the QNN to minimize the loss function, thereby improving the model's accuracy in classifying the data points. The workflow also includes functions to calculate the model's accuracy and loss during training, providing insights into its performance. In summary, this script establishes a quantum machine learning pipeline that leverages quantum circuits for data classification tasks. It highlights the integration of quantum computing concepts with traditional machine learning techniques, demonstrating the potential of quantum algorithms in solving complex classification problems.

## Run the workflow

- python3.8 -m venv .venv
- source .venv/bin/activate
- pip install -r requirements.txt
- covalent start
- python script.py
