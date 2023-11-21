# https://medium.com/@filip_98594/improving-chest-x-ray-pneumonia-detection-with-federated-learning-and-covalent-ff60eef7946c
# https://github.com/AgnostiqHQ/covalent/blob/develop/doc/source/tutorials/federated_learning/source.ipynb

import covalent as ct
import os
import time
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch.optim as optim
from tqdm import tqdm
from torch import nn
import numpy as np
import pandas as pd
import torch
from dataclasses import dataclass
from typing import Tuple, Callable, Dict
import pickle
from collections import Counter


@dataclass
class HFDataset:
    name: Tuple[str, str]

hf_datasets = [
    HFDataset(
        name=("keremberke/chest-xray-classification", 'full'),
    ),
    HFDataset(
        name=("mmenendezg/raw_pneumonia_x_ray", ),
    )
]

class PneumoniaDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        return item['image'], item['label']

class PneumoniaNet(nn.Module):
    """
    Simple CNN for pneumonia detection.
    """
    def __init__(self, image_dim=64):
        super(PneumoniaNet, self).__init__()
        # channel number is 1 for grayscale images
        # use 3 for RGB images
        channel_number = 1
        self.image_dim = image_dim

        self.conv1 = nn.Conv2d(
            in_channels=channel_number, out_channels=16, kernel_size=3,
            stride=1, padding=1
        )
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3,
            stride=1, padding=1
        )
        self.relu2 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(
            kernel_size=2, stride=2
        )

        self.conv3 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3,
            stride=1, padding=1
        )
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3,
            stride=1, padding=1
        )
        self.relu4 = nn.ReLU()
        self.batchnorm1 = nn.BatchNorm2d(128)
        self.maxpool2 = nn.MaxPool2d(
            kernel_size=2, stride=2
        )
        mapping = {
            64: 32768,
            128: 131072
        }
        self.fc1 = nn.Linear(mapping[self.image_dim], 128)
        self.batchnorm2 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool1(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.batchnorm1(x)
        x = self.maxpool2(x)

        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.batchnorm2(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        output = self.sigmoid(x)
        return output

@ct.electron
def create_pneumonia_network(
    image_dimension
):
    return PneumoniaNet(image_dimension)

@ct.electron
def preprocess(dataset, image_transformation):
    dataset["pixel_values"] = [image_transformation(img.convert("RGB")) for img in dataset["image"]]
    del dataset["image"]
    return dataset

@ct.electron
def prepare_dataset(dataset_name, image_dimension=64):
    ds_name_path = dataset_name[0].replace('/', '-')
    save_path = f'preprocessed-obj-{ds_name_path}'

    if len(dataset_name) == 2:
        ds_name, config = dataset_name
    elif len(dataset_name) == 1:
        ds_name = dataset_name[0]
        config = None

    if os.path.exists(save_path):
        print('Loading preprocessed data from local')
        with open(save_path, 'rb') as f:
            preprocessed_train, preprocesed_test = pickle.load(f)
    else:
        print('Preprocessing data')
        if config:
            dataset = load_dataset(ds_name, config)
        else:
            dataset = load_dataset(ds_name)
        image_transformation = transforms.Compose([
            transforms.Resize(
                size=(image_dimension, image_dimension)
            ),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5], std=[0.5]
            )
        ])
        preprocessed_train = preprocess(
            dataset['train'], image_transformation,
        )
        preprocesed_test = preprocess(
            dataset['test'], image_transformation,
        )
        preprocessed_data = (
            preprocessed_train, preprocesed_test
        )
        with open(save_path, "wb") as f:
            pickle.dump(preprocessed_data, f)

    train_ds = PneumoniaDataset(preprocessed_train)
    test_ds = PneumoniaDataset(preprocesed_test)

    return train_ds, test_ds

def create_dataloaders(train_dataset, test_dataset, batch_size):
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_dataloader, test_dataloader

def build_pneumonia_classifier(dataset_name, epoch_number=2, batch_size=64, image_dimension=64):
    train_ds, test_ds = prepare_dataset(
        dataset_name, image_dimension=image_dimension
    )
    
    train_dataloader, test_dataloader = create_dataloaders(
        train_ds, test_ds, batch_size
    )

    model = create_pneumonia_network(image_dimension)

    train_losses, ds_size = train_model(
        model, epoch_number, train_dataloader
    )
    test_acc, test_loss = evaluate(model, test_dataloader)
    return model, ds_size, test_acc

@ct.electron
@ct.lattice
def pnemonia_classifier(**kwargs):
    electron = ct.electron(build_pneumonia_classifier)
    return electron(**kwargs)


@ct.electron
def train_model(model, epoch_count, train_dataloader):
    print("Training model")
    ds_size = len(train_dataloader.dataset)
    losses = []
    optimizer = optim.SGD(model.parameters(), lr=0.00005, momentum=0.9)
    criterion = nn.BCELoss()

    for epoch in range(epoch_count):
        model.train()
        running_loss = 0
        train_correct = 0

        for images, labels in train_dataloader:
            labels = labels.float()
            optimizer.zero_grad()
            output = model(images).flatten()
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            losses.append(loss)
            running_loss += loss.item()
            predicted = (output > 0.5).long()
            train_correct += (predicted == labels).sum().item()

        train_acc = train_correct / ds_size
        print("Epoch {} - Training loss: {:.4f} - Accuracy: {:.4f}".format(
            epoch + 1, running_loss / len(train_dataloader), train_acc)
        )

    return losses, ds_size


@ct.electron
def evaluate(
    model, test_dataloader
):
    criterion = nn.BCELoss()
    model.eval()
    test_loss = 0
    test_correct = 0

    with torch.no_grad():
        for images, labels in test_dataloader:
            labels = labels.float()
            output = model(images).flatten()
            loss = criterion(output, labels)
            test_loss += loss.item()

            predicted = (output > 0.5).long()
            test_correct += (predicted == labels).sum().item()

        test_acc = test_correct / len(test_dataloader.dataset)
        print(
            "Test loss: {:.4f} - Test accuracy: {:.4f}".format(
                test_loss / len(test_dataloader), test_acc
            )
        )

    return test_acc, test_loss

@ct.electron
def create_aggregated_network(model_list, ds_sizes, image_dimension=64):
    dataset_weights = np.array(ds_sizes) / sum(ds_sizes)
    whole_aggregator = []

    for p_index, layer in enumerate(model_list[0].parameters()):
        params_aggregator = torch.zeros(layer.size())

        for model_index, model in enumerate(model_list):
            params_aggregator = params_aggregator + dataset_weights[
                model_index
            ] * list(model.parameters())[p_index].data
        whole_aggregator.append(params_aggregator)

    net_avg = create_pneumonia_network(image_dimension)

    for param_index, layer in enumerate(net_avg.parameters()):
        layer.data = whole_aggregator[param_index]
    return net_avg

@ct.lattice
def federated_workflow(datasets: HFDataset, round_number, epoch_per_round, image_dimension=64):
    test_accuracies = []
    model_showcases = []
    for round_idx in range(round_number):
        models = []
        dataset_sizes = []
        for ds in datasets:
            trained_model, ds_size, test_accuracy = pnemonia_classifier(
                dataset_name=ds.name,
                image_dimension=image_dimension,
                epoch_number=epoch_per_round,
            )

            models.append(trained_model)
            dataset_sizes.append(ds_size)
            test_accuracies.append((round_idx + 1, ds.name, test_accuracy))

            if round_idx == round_number - 1:
                model_showcases.append((trained_model, ds.name))

        model_agg = create_aggregated_network(
            models, dataset_sizes, image_dimension=image_dimension,
        )
        if round_idx == round_number - 1:
            model_showcases.append((model_agg, "aggregated"))

    return test_accuracies, model_showcases

if __name__ == "__main__":
    dispatch_id = ct.dispatch(federated_workflow)(hf_datasets, round_number=1, epoch_per_round=1)
    print(f"\n{dispatch_id}")
    res = ct.get_result(dispatch_id, wait=True)
    print(res)