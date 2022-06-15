from torch import nn
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class Trainer_GCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(500, 256)
        self.conv2 = GCNConv(256, 64)
        self.denseB = Linear(64, 1)
        self.denseW = Linear(64, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        # Linear layer for biases pred
        biases = self.denseB(x)

        # Edge contains sum of adj nodes: https://github.com/pyg-team/pytorch_geometric/discussions/3554
        src, dst = data.edge_index
        src, dst = src[::2], dst[::2]       # skip repeated edges
        weights = (x[src] + x[dst]) / 2
        weights = self.denseW(weights)

        return weights, biases

class Simple_NN(nn.Module):
    def __init__(self):
        super(Simple_NN, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

        # Note: Softmax is not needed for cost calculation
        # Only apply softmax when doing inferencing