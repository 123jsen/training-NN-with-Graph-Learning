### Imports ###
import random
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader

# Imports from custom packages
from utility.model2text import write_design, write_weights, write_metrics
from models.dense_nets import DenseNet
from datasets.class_data import ClassificationDataset


### Constants ###
MAX_DEPTH = 1
LAYER_HEIGHTS = (4, 6, 8, 12, 16, 24, 32, 48, 64, 72, 96, 128)


### Parameters ###
num_epochs = 35
batch_size = 64
test_size = 100     # number of data samples used to calculate accuracy of model

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device\n")


### Functions ###


# Related to Model Generation
def rand_design(num_input=10, num_output=10):
    '''Returns a tuple representing layers height'''
    depth = random.randrange(MAX_DEPTH) + 1
    result = np.asarray([random.choice(LAYER_HEIGHTS) for i in range(depth)])
    return np.concatenate(([num_input], result, [num_output]))


def rand_model(num_input=10, num_output=10):
    '''Returns a random dense neural net model'''
    design = rand_design(num_input, num_output)
    return design, DenseNet(design).to(device)


# Related to Model Training
def train_model(dataloader, model, loss_fn, optimizer):
    '''Function which trains model'''
    model.train()

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def test_model(dataloader, model, loss_fn):
    '''Function which outputs loss and accuracy'''

    with torch.no_grad():
        model.eval()

        X, y = iter(dataloader).next()
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        # accuracy calculation
        pred = np.argmax(pred.cpu(), axis=1)
        y = np.argmax(y.cpu(), axis=1)

        acc = torch.sum(pred == y) / test_size

        return loss, acc


def generate_model_data(num_models=10, target_dir="./dataset_0/"):
    '''Read the data from the csv files, trains a bunch of NNs, and store them at the folder'''

    dataset = ClassificationDataset(dest_dir=target_dir)
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset, batch_size=test_size, shuffle=True)

    input_size, output_size = len(dataset[0][0]), len(dataset[0][1])
    print(
        f"Loaded data from {target_dir}, input = {input_size}, output = {output_size}")
    print(f"Parameters | Epoch: {num_epochs} | Batch Size: {batch_size}")

    for i in range(num_models):
        print()
        print(f"Generating the {i+1}-th model")

        design, model = rand_model(input_size, output_size)
        print(f"Design | {design}")

        write_design(target_dir + "model_designs.txt", design)
        write_weights(target_dir + "model_init_weights.txt",
                      target_dir + "model_init_biases.txt",
                      model)

        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        for epoch in range(num_epochs):
            train_model(train_dataloader, model, loss_fn, optimizer)

        loss, acc = test_model(test_dataloader, model, loss_fn)

        print(f"Training done | loss: {loss:.5f} | acc: {acc:.5f}")

        write_weights(target_dir + "model_weights.txt",
                      target_dir + "model_biases.txt",
                      model)
        write_metrics(target_dir + "model_metrics.txt", loss, acc)

        print("Model data written to file")