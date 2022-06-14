import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from dataset_graphs import graph_from_design

# Constants
SOURCE = "./raw/dataset_0/"

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Data loading from csv
    features = np.genfromtxt(SOURCE + 'data_features.csv', delimiter = ", ")
    targets = np.genfromtxt(SOURCE + 'data_targets.csv', delimiter = ", ")
    train_data = (torch.tensor(features), torch.tensor(targets))

    data = graph_from_design([3, 32, 32, 2])
    data.x = torch.ones(data.num_nodes)

    model = NeuralNetwork().to(device)

    print(data)
