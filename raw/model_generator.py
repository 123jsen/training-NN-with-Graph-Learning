### Imports ###
from asyncio.windows_events import NULL
from msilib.schema import Class
import random
import os
import numpy as np

""" import keras
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.optimizers import Adam
from keras.losses import CategoricalCrossentropy """

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device\n")

### Constants ###
MAX_DEPTH = 5
LAYER_HEIGHTS = (4, 8, 16, 32, 64, 96)


### Parameters ###
num_epochs = 50
batch_size = 32
test_size = 200


### Functions ###

# Related to Model Generation
class Dense_net(nn.Module):
    '''Creates dense NN given a particular design'''

    def __init__(self, design):
        super(Dense_net, self).__init__()
        self.design = design

        # Note: using a for loop will not work by itself, see https://discuss.pytorch.org/t/using-for-loops-in-net-initialization/19817
        self.layers = [None for i in range(len(self.design) - 1)]
        for i in range(len(self.design) - 1):
            self.layers[i] = nn.Linear(design[i], design[i+1])

        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
            x = F.relu(x)
        x = self.layers[-1](x)
        return x


def rand_design(num_input=10, num_output=10):
    '''Returns a tuple representing layers height'''
    depth = random.randrange(MAX_DEPTH) + 1
    result = np.asarray([random.choice(LAYER_HEIGHTS) for i in range(depth)])
    return np.concatenate(([num_input], result, [num_output]))


def rand_model(num_input=10, num_output=10):
    design = rand_design(num_input, num_output)
    return design, Dense_net(design).to(device)


# Related to Model Training
class ClassificationDataset(Dataset):
    def __init__(self, dest_dir):
        self.features = np.genfromtxt(
            dest_dir + 'data_features.csv', delimiter=', ')
        self.features = torch.tensor(self.features, dtype=torch.float32)

        self.targets = np.genfromtxt(
            dest_dir + 'data_targets.csv', delimiter=', ')
        self.targets = torch.tensor(self.targets, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.targets[index]


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


# Related to Saving data
def save_model_data(target_dir, design, model, loss, acc):

    # Design
    with open(target_dir + 'model_designs.txt', 'a') as designs_file:
        np.savetxt(designs_file, design, fmt="%d, ", newline="")
        designs_file.write("\n")

    # Metadata
    if not(os.path.isfile(target_dir + 'model_meta.txt')):
        with open(target_dir + 'model_meta.txt', 'w') as meta_file:
            meta_file.write("loss, accuracy")
            meta_file.write("\n")

    with open(target_dir + 'model_meta.txt', 'a') as meta_file:
        meta_file.write(f"{loss}, {acc}\n")

    # Weights and Biases
    for i in range(len(model.layers)):
        weights = model.state_dict()[f"layers.{i}.weight"].cpu()
        with open(target_dir + 'model_weights.txt', 'a') as weights_file:
            np.savetxt(weights_file, weights.T, delimiter=", ")

        biases = model.state_dict()[f"layers.{i}.bias"].cpu()
        with open(target_dir + 'model_biases.txt', 'a') as biases_file:
            np.savetxt(biases_file, biases.T, newline=", ")
            biases_file.write("\n")


def generate_model_data(num_models=10, target_dir="./dataset_1/"):
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

        # This part trains the model
        design, model = rand_model(input_size, output_size)
        print(f"Design = {design}")

        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        for epoch in range(num_epochs):
            train_model(train_dataloader, model, loss_fn, optimizer)

        loss, acc = test_model(test_dataloader, model, loss_fn)

        print(f"Training done | loss: {loss:.5f} | acc: {acc:.5f}")

        save_model_data(target_dir, design, model, loss, acc)

        print("Model data written to file")


### Main Function ###
if __name__ == "__main__":
    generate_model_data(num_models=10)
