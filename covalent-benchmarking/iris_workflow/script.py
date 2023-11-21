# https://github.com/AgnostiqHQ/covalent/blob/develop/doc/source/tutorials/1_QuantumMachineLearning/pennylane_iris_classification/source.ipynb

import time
import pennylane as qml
from pennylane import numpy as np
import covalent as ct
import matplotlib.pyplot as plt
from pennylane.optimize import NesterovMomentumOptimizer
import csv
import psutil
import itertools
from typing import Any, List, Optional, Tuple


@ct.electron
def get_angles(x):
    beta0 = 2 * np.arcsin(np.sqrt(x[1] ** 2) / np.sqrt(x[0] ** 2 + x[1] ** 2 + 1e-12))
    beta1 = 2 * np.arcsin(np.sqrt(x[3] ** 2) / np.sqrt(x[2] ** 2 + x[3] ** 2 + 1e-12))
    beta2 = 2 * np.arcsin(
        np.sqrt(x[2] ** 2 + x[3] ** 2) / np.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + x[3] ** 2)
    )
    return np.array([beta2, -beta1 / 2, beta1 / 2, -beta0 / 2, beta0 / 2])


def statepreparation(a):
    qml.RY(a[0], wires=0)
    qml.CNOT(wires=[0, 1])
    qml.RY(a[1], wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RY(a[2], wires=1)
    qml.PauliX(wires=0)
    qml.CNOT(wires=[0, 1])
    qml.RY(a[3], wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RY(a[4], wires=1)
    qml.PauliX(wires=0)


def layer(W):
    qml.Rot(W[0, 0], W[0, 1], W[0, 2], wires=0)
    qml.Rot(W[1, 0], W[1, 1], W[1, 2], wires=1)
    qml.CNOT(wires=[0, 1])


@qml.qnode(qml.device("default.qubit", wires=2))
def circuit(weights, angles):
    statepreparation(angles)
    for W in weights:
        layer(W)
    return qml.expval(qml.PauliZ(0))


@ct.electron
def square_loss(labels, predictions):
    loss = 0
    for l, p in zip(labels, predictions):
        loss = loss + (l - p) ** 2

    loss = loss / len(labels)
    return loss


@ct.electron
def variational_classifier(weights, bias, angles):
    return circuit(weights, angles) + bias


@ct.electron
def cost(weights, bias, features, labels):
    predictions = [variational_classifier(weights, bias, f) for f in features]
    return square_loss(labels, predictions)


@ct.electron
def load_features(data):
    X = data[:, 0:2]
    padding = 0.3 * np.ones((len(X), 1))
    X_pad = np.c_[np.c_[X, padding], np.zeros((len(X), 1))]
    normalization = np.sqrt(np.sum(X_pad**2, -1))
    X_norm = (X_pad.T / normalization).T
    features = np.array([get_angles(x) for x in X_norm], requires_grad=False)
    Y = data[:, -1]
    return features, Y, X, X_norm, X_pad


@ct.electron
def train_val_split(features, Y):
    np.random.seed(0)
    num_data = len(Y)
    num_train = int(0.75 * num_data)
    index = np.random.permutation(range(num_data))
    feats_train = features[index[:num_train]]
    Y_train = Y[index[:num_train]]
    feats_val = features[index[num_train:]]
    Y_val = Y[index[num_train:]]
    return feats_train, Y_train, feats_val, Y_val, index, num_train


@ct.electron
def accuracy(labels, predictions):
    loss = 0
    for l, p in zip(labels, predictions):
        if abs(l - p) < 1e-5:
            loss = loss + 1
    loss = loss / len(labels)
    return loss


@ct.electron
def weights_bias_init(num_layers, num_qubits):
    weights_init = 0.01 * np.random.randn(num_layers, num_qubits, 3, requires_grad=True)
    bias_init = np.array(0.0, requires_grad=True)
    return weights_init, bias_init


@ct.electron
def training(
    iterations,
    batch_size,
    weights,
    bias,
    num_train,
    feats_train,
    Y_train,
    opt,
    feats_val,
    Y_val,
    Y,
):
    start_time = time.time()
    start_cpu = psutil.cpu_percent(4)
    start_memory_usage = psutil.virtual_memory().percent

    training_steps = []
    accuracy_steps_train = []
    accuracy_steps_val = []
    weights_init = weights
    bias_init = bias
    for it in range(iterations):
        batch_index = np.random.randint(0, num_train, (batch_size,))
        feats_train_batch = feats_train[batch_index]
        Y_train_batch = Y_train[batch_index]
        weights_init, bias_init, _, _ = opt.step(
            cost, weights_init, bias_init, feats_train_batch, Y_train_batch
        )
        training_steps.append(it)
        predictions_train = [
            np.sign(variational_classifier(weights_init, bias_init, f)) for f in feats_train
        ]
        predictions_val = [
            np.sign(variational_classifier(weights_init, bias_init, f)) for f in feats_val
        ]
        acc_train = accuracy(Y_train, predictions_train)
        acc_val = accuracy(Y_val, predictions_val)
        accuracy_steps_train.append(acc_train)
        accuracy_steps_val.append(acc_val)
    
    metrics = {
        "time (seconds)": time.time() - start_time,
        "cpu (%)": (psutil.cpu_percent(4) + start_cpu) / 2,
        "memory (%)": (psutil.virtual_memory().percent + start_memory_usage) / 2,
    }

    return metrics


@ct.lattice
def workflow(
    data, iterations, num_layers, num_qubits, batch_size
):
    features, Y, _, _, _ = load_features(data)
    feats_train, Y_train, feats_val, Y_val, _, num_train = train_val_split(features, Y)

    opt = NesterovMomentumOptimizer(0.005)
    weights, bias = weights_bias_init(num_layers, num_qubits)
    metrics = training(
        iterations=iterations,
        batch_size=batch_size,
        weights=weights,
        bias=bias,
        num_train=num_train,
        feats_train=feats_train,
        Y_train=Y_train,
        opt=opt,
        feats_val=feats_val,
        Y_val=Y_val,
        Y=Y,
    )
    return metrics

def run_workflow(
        range_iterations: List[int] = [100, 200],
        range_layers: List[int] = [1, 2, 4, 6],
        range_qubits: List[int] = [2, 4, 6],
        range_batch_sizes: List[int] = [16, 32],
):
    all_results = []
    data = np.loadtxt("data/iris_classes1and2_scaled.txt")
    
    for n_iterations, n_layers, n_qubits, batch_size in itertools.product(range_iterations, range_layers, range_qubits, range_batch_sizes):
        print(f"Running workflow with: iterations={n_iterations}, layers={n_layers}, qubits={n_qubits}, batch_size={batch_size}")

        dispatch_id = ct.dispatch(workflow)(
            data=data,
            iterations=n_iterations,
            num_layers=n_layers,
            num_qubits=n_qubits,
            batch_size=batch_size
        )

        result = ct.get_result(dispatch_id, wait=True)

        if result.status == 'FAILED':
            print("Workflow failed, something went wrong")
            all_results.append({
                    'id': dispatch_id,
                    'config': [n_iterations, n_layers, n_qubits, batch_size],
                    'error': 'Workflow failed'
            })
        else:
            print(f"Workflow finished with result: {result.result}")

            all_results.append({
                'id': dispatch_id,
                'config': [n_iterations, n_layers, n_qubits, batch_size],
                'metrics': result.result
            })

    return all_results


if __name__ == "__main__":
    results = run_workflow()
    
    with open('metrics.csv', mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Dispatch ID', 'Iterations', 'Layers', 'Qubits', 'Batch Size', 'Metrics'])
        for result in results:
            config = result['config']
            metrics = result['metrics']
            writer.writerow([
                result['id'], 
                config[0], config[1], config[2], config[3],
                metrics
            ])

    print(f"Results written to metrics.csv")