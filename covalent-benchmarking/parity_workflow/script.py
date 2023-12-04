# https://github.com/AgnostiqHQ/covalent/blob/develop/doc/source/tutorials/1_QuantumMachineLearning/pennylane_parity_classifier/source.ipynb

import pennylane as qml
from pennylane import numpy as np
import covalent as ct
from pennylane.optimize import NesterovMomentumOptimizer
from typing import List
import itertools
import time
import psutil
import csv

dev = qml.device("default.qubit", wires=4)

def layer(W):
    qml.Rot(W[0, 0], W[0, 1], W[0, 2], wires=0)
    qml.Rot(W[1, 0], W[1, 1], W[1, 2], wires=1)
    qml.Rot(W[2, 0], W[2, 1], W[2, 2], wires=2)
    qml.Rot(W[3, 0], W[3, 1], W[3, 2], wires=3)
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    qml.CNOT(wires=[2, 3])
    qml.CNOT(wires=[3, 0])

def statepreparation(x):
    qml.BasisState(x, wires=[0, 1, 2, 3])


@qml.qnode(dev)
def circuit(weights, x):
    statepreparation(x)
    for W in weights:
        layer(W)
    return qml.expval(qml.PauliZ(0))

def variational_classifier(weights, bias, x):
    return circuit(weights, x) + bias

def square_loss(labels, predictions):
    loss = 0
    for l, p in zip(labels, predictions):
        loss = loss + (l - p) ** 2
    loss = loss / len(labels)
    return loss


def accuracy(labels, predictions):
    loss = 0
    for l, p in zip(labels, predictions):
        if abs(l - p) < 1e-5:
            loss = loss + 1
    loss = loss / len(labels)
    return loss


def cost(weights, bias, X, Y):
    predictions = [variational_classifier(weights, bias, x) for x in X]
    return square_loss(Y, predictions)

@ct.electron
def get_optimizer():
    return NesterovMomentumOptimizer(0.25)


@ct.electron
def weights_bias_init(num_layers, num_qubits):
    weights_init = 0.01 * np.random.randn(num_layers, num_qubits, 3, requires_grad=True)
    bias_init = np.array(0.0, requires_grad=True)
    return weights_init, bias_init

@ct.electron
def training(opt, weights, bias, epochs, batch_size, X, Y, num_layers, num_qubits, cost):
    training_steps = []
    cost_steps = []
    accuracy_steps = []
    for it in range(epochs):
        batch_index = np.random.randint(0, len(X), (batch_size,))
        X_batch = X[batch_index]
        Y_batch = Y[batch_index]
        weights, bias, _, _ = opt.step(cost, weights, bias, X_batch, Y_batch)
        predictions = [np.sign(variational_classifier(weights, bias, x)) for x in X]
        acc = accuracy(Y, predictions)
        training_steps.append(it)
        cost_steps.append(cost(weights, bias, X, Y))
        accuracy_steps.append(acc)

        print(
            "Iter: {:5d} | Cost: {:0.7f} | Accuracy: {:0.7f} ".format(
                it + 1, cost(weights, bias, X, Y), acc
            )
        )

    return weights, bias, training_steps, cost_steps, accuracy_steps


@ct.lattice
def workflow(epochs, num_layers, num_qubits, X, Y):
    start_time = time.time()
    start_cpu = psutil.cpu_percent(4)
    start_memory_usage = psutil.virtual_memory().percent

    opt = get_optimizer()
    weights, bias = weights_bias_init(num_layers, num_qubits)
    batch_size = 5
    weights, bias, training_steps, cost_steps, accuracy_steps = training(
        opt, weights, bias, epochs, batch_size, X, Y, num_layers, num_qubits, cost
    )

    metrics = {
        "time (seconds)": time.time() - start_time,
        "cpu (%)": (psutil.cpu_percent(4) + start_cpu) / 2,
        "memory (%)": (psutil.virtual_memory().percent + start_memory_usage) / 2,
    }

    return metrics

def run_workflow(
    range_epochs: List[int] = [2, 4, 6, 8],
    range_layers: List[int] = [5, 10, 15, 20, 25],
    range_qubits: List[int] = [4, 6, 8, 10, 12],
):
    all_results = []

    data = np.loadtxt("parity.txt")
    X = np.array(data[:, :-1], requires_grad=False)
    Y = np.array(data[:, -1], requires_grad=False)
    Y = Y * 2 - np.ones(len(Y))

    for n_epochs, n_layers, n_qubits in itertools.product(range_epochs, range_layers, range_qubits):
        print(f"Running workflow with: epochs={n_epochs}, layers={n_layers}, qubits={n_qubits}")

        dispatch_id = ct.dispatch(workflow)(
            epochs=n_epochs,
            num_layers=n_layers,
            num_qubits=n_qubits,
            X=X,
            Y=Y
        )

        result = ct.get_result(dispatch_id, wait=True)

        if result.status == 'FAILED':
            print("Workflow failed, something went wrong")
            return
        else:
            print(f"Workflow finished with result: {result.result}")

            all_results.append({
                'id': dispatch_id,
                'config': [n_epochs, n_layers, n_qubits],
                'metrics': result.result
            })

    return all_results




if __name__ == "__main__":
    results = run_workflow()

    with open('metrics.csv', mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Dispatch ID', 'Epochs', 'Layers', 'Qubits', 'Metrics'])
        for result in results:
            config = result['config']
            writer.writerow([
                result['id'], 
                config[0], config[1], config[2], 
                result['metrics'],
            ])

    print(f"Results written to metrics.csv")