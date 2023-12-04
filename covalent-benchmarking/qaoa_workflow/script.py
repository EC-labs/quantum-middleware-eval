# https://github.com/AgnostiqHQ/covalent/blob/develop/doc/source/tutorials/1_QuantumMachineLearning/qaoa_maxcut/source.ipynb

import pennylane as qml
from pennylane import qaoa
import numpy as np
import networkx as nx
import covalent as ct
from typing import List
import itertools
import time
import psutil
import csv


@ct.electron
def make_graph(qubits, prob):
    graph = nx.generators.random_graphs.gnp_random_graph(n=qubits, p=prob)
    cost_h, mixer_h = qaoa.min_vertex_cover(graph, constrained=False)
    return cost_h, mixer_h


@ct.electron
def get_circuit(cost_h, mixer_h):
    def qaoa_layer(gamma, alpha):
        qaoa.cost_layer(gamma, cost_h)
        qaoa.mixer_layer(alpha, mixer_h)

    def circuit(params, wires, **kwargs):
        depth = params.shape[1]
        for w in range(wires):
            qml.Hadamard(wires=w)
        qml.layer(qaoa_layer, depth, params[0], params[1])

    return circuit


@ct.electron
def make_cost_function(circuit, cost_h, qubits):
    dev = qml.device("lightning.qubit", wires=qubits)

    @qml.qnode(dev)
    def cost_function(params):
        circuit(params, wires=qubits)
        return qml.expval(cost_h)

    return cost_function


@ct.electron
def get_random_initialization(p=1):
    return np.random.uniform(0, 2 * np.pi, (2, p), requires_grad=True)


@ct.electron
def initialize_parameters(p=1, qubits=2, prob=0.3):
    cost_h, mixer_h = make_graph(qubits=qubits, prob=prob)
    circuit = get_circuit(cost_h, mixer_h)
    cost_function = make_cost_function(circuit, cost_h, qubits)
    initial_angles = get_random_initialization(p=p)
    return cost_function, initial_angles


@ct.electron
def calculate_cost(cost_function, params, optimizer):
    params, loss = optimizer.step_and_cost(cost_function, params)
    return optimizer, params, loss


@ct.electron
def optimize_electron(
    cost_function, init_angles, optimizer=qml.GradientDescentOptimizer(), iterations=10
):
    start_time = time.time()
    start_cpu = psutil.cpu_percent(4)
    start_memory_usage = psutil.virtual_memory().percent

    loss_history = []
    params = init_angles
    for _ in range(iterations):
        optimizer, params, loss = calculate_cost(
            cost_function=cost_function, params=params, optimizer=optimizer
        )
        loss_history.append(loss)
    
    metrics = {
        "time (seconds)": time.time() - start_time,
        "cpu (%)": (psutil.cpu_percent(4) + start_cpu) / 2,
        "memory (%)": (psutil.virtual_memory().percent + start_memory_usage) / 2,
    }

    return loss_history, metrics


@ct.electron
def collect_and_mean(array):
    return np.mean(array, axis=0)


@ct.electron
def get_random_initialization(p=1, lower_l=0, higher_l=2 * np.pi):
    return np.random.uniform(lower_l, higher_l, (2, p))


@ct.electron
def initialize_parameters(p=1, qubits=2, prob=0.3, lower_l=0, higher_l=2 * np.pi):
    cost_h, mixer_h = make_graph(qubits=qubits, prob=prob)
    circuit = get_circuit(cost_h, mixer_h)
    cost_function = make_cost_function(circuit, cost_h, qubits)
    initial_angles = get_random_initialization(p=p, lower_l=lower_l, higher_l=higher_l)
    return cost_function, initial_angles


@ct.lattice
def workflow(
    p=1,
    qubits=2,
    prob=0.3,
    optimizers=[qml.GradientDescentOptimizer()],
    iterations=10,
    limits=[[0, 2 * np.pi]],
):
    compare_optimizers = []
    tmp = []
    for lower_l, higher_l in limits:
        cost_function, init_angles = initialize_parameters(
            p=p, qubits=qubits, prob=prob, lower_l=lower_l, higher_l=higher_l
        )
        for optimizer in optimizers:
            loss_history, metrics = optimize_electron(
                cost_function=cost_function,
                init_angles=init_angles,
                optimizer=optimizer,
                iterations=iterations,
            )
            tmp.append(loss_history)
    compare_optimizers.append(tmp)
    return compare_optimizers, metrics


def run_workflow(
    range_qubits: List[int] = [2],
    range_iterations: List[int] = [10],
):
    all_results = []

    for n_qubits, n_iterations in itertools.product(range_qubits, range_iterations):
        print(f"Running workflow with: qubits={n_qubits}, iterations={n_iterations}")

        dispatch_id = ct.dispatch(workflow)(
            p=1,
            prob=0.3,
            optimizers=[
            qml.GradientDescentOptimizer(),
            qml.AdagradOptimizer(),
            qml.MomentumOptimizer(),
            qml.AdamOptimizer(),
            ],
            iterations=n_iterations,
            qubits=n_qubits,
            limits=[
            [0, np.pi / 2],
            [np.pi / 2, np.pi],
            [np.pi, np.pi * 1.5],
            [np.pi * 1.5, 2 * np.pi],
            ]
        )

        result = ct.get_result(dispatch_id, wait=True)

        if result.status == 'FAILED':
            print("Workflow failed, something went wrong")
            return
        else:
            print(f"Workflow finished with result: {result.result[-1]}")

            all_results.append({
                'id': dispatch_id,
                'config': [n_qubits, n_iterations],
                'metrics': result.result[-1]
            })

    return all_results


if __name__ == "__main__":
    results = run_workflow()

    with open('metrics.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Dispatch ID', 'Qubits', 'Iterations', 'Metrics'])
        for result in results:
            writer.writerow([result['id'], result['config'][0], result['config'][1], result['metrics']])
    
    print(f"Results written to metrics.csv")