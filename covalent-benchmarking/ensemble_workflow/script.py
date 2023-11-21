"""
Link: https://medium.com/qiskit/hybrid-quantum-workflows-simplified-with-covalent-aws-and-ibm-quantum-d6202c11bd01

Hybrid classifier based on:
    https://towardsdatascience.com/binary-image-classification-in-pytorch-5adf64f8c781
by Marcello Politi

Data source:
    https://www.kaggle.com/datasets/biaiscience/dogs-vs-cats
"""

import os
import time
import itertools
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Tuple
from zipfile import ZipFile
import psutil
import csv
import covalent as ct
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from torch import Tensor, nn
from torch.nn.modules.loss import L1Loss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet34

_BASE_PATH = Path(__file__).parent.resolve()
DATA_DIR = _BASE_PATH / "data"

metrics_file = _BASE_PATH / 'metrics.csv'

class ParametricQC:
    """simplify interface for getting expectation value from quantum circuit"""
    RETRY_MAX: int = 5
    runs_total: int = 0
    calls_total: int = 0

    def __init__(
        self,
        n_qubits: int,
        shift: float,
        estimator: Estimator,
    ):
        self.n_qubits = n_qubits
        self.shift = shift
        self.estimator = estimator
        self._init_circuit_and_observable()

    def _init_circuit_and_observable(self):
        qr = QuantumRegister(size=self.n_qubits)

        self.circuit = QuantumCircuit(qr)
        self.circuit.barrier()
        self.circuit.h(range(self.n_qubits))
        self.thetas = []
        for i in range(self.n_qubits):
            theta = Parameter(f"theta{i}")
            self.circuit.ry(theta, i)
            self.thetas.append(theta)

        self.circuit.assign_parameters({theta: 0.0 for theta in self.thetas})
        self.obs = SparsePauliOp("Z" * self.n_qubits)

    def run(self, inputs: Tensor) -> Tensor:
        """use inputs as parameters to compute expectation"""

        parameter_values = inputs.tolist()
        circuits_batch = [self.circuit] * len(parameter_values)
        observables = [self.obs] * len(parameter_values)
        exps = self._run(parameter_values, circuits_batch, observables).result()
        return torch.tensor(exps.values).unsqueeze(dim=0).T

    def _run(
        self,
        parameter_values: List[Any],
        circuits: List[QuantumCircuit],
        observables: List[SparsePauliOp],
    ):

        # run job inside a try-except loop and retry if something goes wrong
        job = None
        retries = 0
        while retries < ParametricQC.RETRY_MAX:

            try:
                job = self.estimator.run(
                    circuits=circuits,
                    observables=observables,
                    parameter_values=parameter_values
                )
                break

            except RuntimeError as re:
                warnings.warn(
                    f"job failed on attempt {retries + 1}:\n\n'{re}'\nresubmitting...",
                    category=UserWarning
                )
                retries += 1

            finally:
                ParametricQC.runs_total += len(circuits)
                ParametricQC.calls_total += 1

        if job is None:
            raise RuntimeError(f"job failed after {retries + 1} retries")
        return job


class QuantumFunction(torch.autograd.Function):
    """custom autograd function that uses a quantum circuit"""

    @staticmethod
    def forward(
        ctx,
        batch_inputs: Tensor,
        qc: ParametricQC,
    ) -> Tensor:
        """forward pass computation"""
        ctx.save_for_backward(batch_inputs)
        ctx.qc = qc
        return qc.run(batch_inputs)

    @staticmethod
    def backward(
        ctx,
        grad_output: Tensor
    ):
        """backward pass computation using parameter shift rule"""
        batch_inputs = ctx.saved_tensors[0]
        qc = ctx.qc

        shifted_inputs_r = torch.empty(batch_inputs.shape)
        shifted_inputs_l = torch.empty(batch_inputs.shape)

        # loop over each input in the batch
        for i, _input in enumerate(batch_inputs):

            # loop entries in each input
            for j in range(len(_input)):

                # compute parameters for parameter shift rule
                d = torch.zeros(_input.shape)
                d[j] = qc.shift
                shifted_inputs_r[i, j] = _input + d
                shifted_inputs_l[i, j] = _input - d

        # run gradients in batches
        exps_r = qc.run(shifted_inputs_r)
        exps_l = qc.run(shifted_inputs_l)

        return (exps_r - exps_l).float() * grad_output.float(), None, None


class QuantumLayer(torch.nn.Module):
    """a neural network layer containing a quantum function"""

    def __init__(
        self,
        n_qubits: int,
        estimator: Estimator,
    ):
        super().__init__()
        self.qc = ParametricQC(
            n_qubits=n_qubits,
            shift=torch.pi / 2,
            estimator=estimator,
        )

    def forward(self, xs: Tensor) -> Tensor:
        """forward pass computation"""

        result = QuantumFunction.apply(xs, self.qc)

        if xs.shape[0] == 1:
            return result.view((1, 1))
        return result

    @property
    def qc_counts(self) -> dict:
        return {
            "n_qubits": self.qc.n_qubits,
            "runs_total": ParametricQC.runs_total,
            "calls_total": ParametricQC.calls_total
        }


def _get_model(n_qubits: int):    
    resnet_model = resnet34()
    resnet_model.fc = nn.Linear(resnet_model.fc.in_features, n_qubits)

    model = nn.Sequential(
        resnet_model,
        QuantumLayer(n_qubits, Estimator()),
    )

    model.to("cuda" if torch.cuda.is_available() else "cpu")
    return model


def _get_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def _dataloader(
    kind: str,
    batch_size: int,
    image_size: int,
    base_dir: Optional[Path] = None,
    shuffle: bool = True,
) -> DataLoader:
    
    print("Loading the data")

    transform = _get_transform(image_size)
    if base_dir is None:
        base_dir = Path(".").resolve()

    def _g(x):
        # rescales target labels from {0,1} to {-1,1}
        return 2 * x - 1

    if kind == "train":
        train_dir = base_dir / DATA_DIR / "train"
        return DataLoader(
            ImageFolder(train_dir, transform=transform, target_transform=_g),
            shuffle=shuffle,
            batch_size=batch_size,
        )

    if kind == "test":
        test_dir = base_dir / DATA_DIR / "test"
        return DataLoader(
            ImageFolder(test_dir, transform=transform, target_transform=_g),
            shuffle=shuffle,
            batch_size=batch_size
        )
    raise ValueError("parameter `kind` must be 'train' or 'test'.")

@dataclass
class TrainingResult:
    backend_name: str
    n_qubits: int
    n_shots: int
    n_epochs: int
    batch_size: int
    image_size: int
    learning_rate: float
    runs_total: int
    calls_total: int
    n_tested: int = 0
    n_correct: int = 0
    losses: List[float] = field(repr=False, default_factory=list)
    epoch_losses: List[float] = field(repr=False, default_factory=list)

@ct.electron()
def train_model(
    n_qubits: int,
    n_shots: int,
    n_epochs: int,
    batch_size: int,
    image_size: int,
    learning_rate: float,
    base_dir: Optional[Path] = None,
) -> TrainingResult:
    start_time = time.time()
    start_cpu = psutil.cpu_percent(4)
    start_memory_usage = psutil.virtual_memory().percent
    
    if not DATA_DIR.exists():
        with ZipFile(f"{DATA_DIR}.zip", "r") as zipped_file:
            zipped_file.extractall()

    losses = []
    epoch_losses = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _get_model(n_qubits)
    loader_train = _dataloader("train", batch_size, image_size, base_dir=base_dir)
    loss_fn = L1Loss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    def _compute_loss(x, y):
        optimizer.zero_grad()
        yhat = model(x)
        model.train()
        loss = loss_fn(yhat, y)
        loss.backward()
        optimizer.step()
        return yhat, loss

    for epoch in range(n_epochs):
        epoch_loss = 0.0

        N = len(loader_train)
        for i, data in enumerate(loader_train):
            x_batch, y_batch = data
            x_batch = x_batch.to(device)
            y_batch = y_batch.unsqueeze(1).float()
            y_batch = y_batch.to(device)

            _, loss = _compute_loss(x_batch, y_batch)

            _loss = loss.item()
            epoch_loss += _loss / N
            losses.append(_loss)

        epoch_losses.append(epoch_loss)

    qc_counts = model[-1].qc_counts

    metrics = {
        "time (seconds)": time.time() - start_time,
        "cpu (%)": (psutil.cpu_percent(4) + start_cpu) / 2,
        "memory (%)": (psutil.virtual_memory().percent + start_memory_usage) / 2,
    }

    return TrainingResult(
        backend_name="local_simulator",
        n_qubits=n_qubits,
        n_shots=n_shots,
        n_epochs=n_epochs,
        batch_size=batch_size,
        image_size=image_size,
        learning_rate=learning_rate,
        runs_total=qc_counts["runs_total"],
        calls_total=qc_counts["calls_total"],
        losses=losses,
        epoch_losses=epoch_losses,
    ), metrics


@ct.electron()
def plot_predictions(
    tr: TrainingResult,
    grid_dims: Tuple[int, int] = (6, 6),
    device: str = "cpu",
    random_seed: Optional[int] = None,
) -> TrainingResult:
    start_time = time.time()
    start_cpu = psutil.cpu_percent(4)
    start_memory_usage = psutil.virtual_memory().percent

    mpl.use(backend="Agg")

    model = _get_model(n_qubits=tr.n_qubits)
    model.to(device)

    if random_seed is not None:
        torch.random.manual_seed(random_seed)

    n = 0
    n_correct = 0
    loader_test = _dataloader(
        "test",
        batch_size=1,
        image_size=tr.image_size,
        base_dir=_BASE_PATH,
    )

    with torch.no_grad():
        model.eval()
        for x, y in loader_test:
            if n >= grid_dims[0] * grid_dims[1]:
                break
            i = n // grid_dims[0]
            j = n % grid_dims[1]

            pred = model(x)
            y_pred = pred.sign()
            if y_pred == y:
                n_correct += 1

            n += 1

    tr.n_tested = n
    tr.n_correct = n_correct

    metrics = {
        "time (seconds)": time.time() - start_time,
        "cpu (%)": (psutil.cpu_percent(4) + start_cpu) / 2,
        "memory (%)": (psutil.virtual_memory().percent + start_memory_usage) / 2,
    }

    return tr, metrics


@ct.lattice
def workflow(
    n_qubits: int = 1,
    n_shots: int = 2,
    n_epochs: int = 1,
    batch_size: int = 16,
    image_size: int = 244,
    learning_rate: float = 1e-4,
) -> TrainingResult:
    
    training_result, metrics_train_model = train_model(
        n_qubits=n_qubits,
        n_shots=n_shots,
        n_epochs=n_epochs,
        batch_size=batch_size,
        image_size=image_size,
        learning_rate=learning_rate,
        base_dir=None,
    )
    
    training_result, metrics_plot_predictions = plot_predictions(training_result)

    metrics = {
        'train_model': metrics_train_model,
        'plot_predictions': metrics_plot_predictions
    }

    return training_result, metrics


def run_workflow(
    range_qubits: List[int] = [1, 2],
    range_shots: List[int] = [100, 200],
    range_epochs: List[int] = [1, 2],
    range_batch_sizes: List[int] = [16, 32],
    range_learning_rates: List[float] = [1e-4, 1e-3]
):
    all_results = []

    for n_qubits, n_shots, n_epochs, batch_size, learning_rate in itertools.product(range_qubits, range_shots, range_epochs, range_batch_sizes, range_learning_rates):
        print(f"Running workflow with: qubits={n_qubits}, shots={n_shots}, epochs={n_epochs}, batch_size={batch_size}, learning_rate={learning_rate}")

        dispatch_id = ct.dispatch(workflow)(
            n_qubits=n_qubits,
            n_shots=n_shots,
            n_epochs=n_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate
        )

        result = ct.get_result(dispatch_id, wait=True)

        if result.status == 'FAILED':
            print("Workflow failed, something went wrong")
            return
        else:
            print(f"Workflow finished with result: {result.result}")

            all_results.append({
                'id': dispatch_id,
                'config': [range_qubits, range_shots, range_epochs, range_batch_sizes, range_learning_rates],
                'metrics': result.result[-1]
            })

    return all_results


if __name__ == "__main__":
    results = run_workflow()
    
    with open('metrics.csv', mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Dispatch ID', 'Qubits', 'Shots', 'Epochs', 'Batch Size', 'Learning Rate', 'Training Metrics', 'Prediction Metrics'])
        for result in results:
            config = result['config']
            metrics_train = result['metrics']['train_model']
            metrics_pred = result['metrics']['plot_predictions']
            writer.writerow([
                result['id'], 
                config[0], config[1], config[2], config[3], config[4], 
                metrics_train,
                metrics_pred
            ])

    print(f"Results written to metrics.csv")
    