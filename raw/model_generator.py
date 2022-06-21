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
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

### Constants ###
MAX_DEPTH = 5
LAYER_HEIGHTS = (4, 8, 16, 32, 64, 96)


### Parameters ###
num_epochs = 250
batch_size = 32


### Functions ###

# Related to Model Generation
class Dense_net(nn.Module):
    '''Creates dense NN given a particular design'''

    def __init__(self, design):
        super(Dense_net, self).__init__()
        self.design = design

        self.layers = [None for i in range(self.design - 1)]
        for i in range(len(self.design) - 1):
            self.layers[i] = nn.Linear(design[i], design[i+1])

    def forward(self, x):
        for i in range(len(self.design) - 1):
            x = self.layers[i](x)
        return x


def rand_design(num_input=10, num_output=10):
    '''Returns a tuple representing layers height'''
    depth = random.randrange(MAX_DEPTH) + 1
    result = np.asarray([random.choice(LAYER_HEIGHTS) for i in range(depth)])
    return np.concatenate(([num_input], result, [num_output]))


def rand_model(num_input=10, num_output=10):
    design = rand_design(num_input, num_output)
    return design, Dense_net(design)


# Related to Model Training
class ClassificationDataset(Dataset):
    def __init__(self, dest_dir):
        self.features = np.genfromtxt(
            dest_dir + 'data_features.csv', delimiter=', ')
        self.targets = np.genfromtxt(
            dest_dir + 'data_targets.csv', delimiter=', ')

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.targets[index]


def train_model(dataloader, model, loss_fn, optimizer):
    '''Function which trains model and returns loss and accuracy'''
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

    # TODO: return final loss and accuracy
    return 1, 1


# Related to Saving data
def save_model_data(target_dir, design, model, loss, accuracy):
    # This part appends the model to text file
    # Design
    with open(target_dir + 'model_designs.txt', 'a') as designs_file:
        np.savetxt(designs_file, design, fmt="%d, ", newline="")
        designs_file.write("\n")

    for layer in model.layers:
        weights = layer.get_weights()
        # Weights
        with open(target_dir + 'model_weights.txt', 'a') as weights_file:
            np.savetxt(weights_file, weights[0], delimiter=", ")

        # Biases
        with open(target_dir + 'model_biases.txt', 'a') as biases_file:
            np.savetxt(biases_file, weights[1], newline=", ")
            biases_file.write("\n")

    # Metadata
    if not(os.path.isfile(target_dir + 'model_meta.txt')):
        with open(target_dir + 'model_meta.txt', 'w') as meta_file:
            meta_file.write("loss, accuracy")
            meta_file.write("\n")

    with open(target_dir + 'model_meta.txt', 'a') as meta_file:
        meta_file.write(
            f"{history.history['loss'][-1]}, {history.history['accuracy'][-1]}\n")


def generate_model_data(num_models=10, target_dir="./dataset_1/"):
    '''Read the data from the csv files, trains a bunch of NNs, and store them at the folder'''

    dataset = ClassificationDataset(dest_dir=target_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    input_size, output_size = len(dataset[0].features[0]), len(dataset[0].targets[0])

    for i in range(num_models):

        # This part trains the model
        design, model = rand_model(input_size, output_size)

        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        for epoch in range(num_epochs):
            loss, acc = train_model(dataloader, model, loss_fn, optimizer)

        save_model_data(target_dir, design, model, loss, acc)

        print("Model data written to file")


### Main Function ###
if __name__ == "__main__":
    generate_model_data(num_models=10)
