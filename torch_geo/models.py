import torch
from torch import nn
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class Trainer_GCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GATConv(in_channels=503, out_channels=32, edge_dim=1)
        # self.conv2 = GATConv(in_channels=256, out_channels=64, edge_dim=1)
        self.denseB = Linear(32, 1)
        self.denseW = Linear(32, 1)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_weight

        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        # x = F.dropout(x, training=self.training)

        # x = self.conv2(x, edge_index, edge_attr)
        # x = F.relu(x)
        # x = F.dropout(x, training=self.training)

        # Linear layer for biases pred
        biases = self.denseB(x)
        # Set all biases on input layer to zero
        biases *= data.input_mask

        # Edge contains sum of adj nodes: https://github.com/pyg-team/pytorch_geometric/discussions/3554
        src, dst = data.edge_index
        if not(data.is_directed()):
            src, dst = src[::2], dst[::2]       # skip repeated edges for undirected graph
        weights = (x[src] + x[dst]) / 2
        weights = self.denseW(weights)

        return weights, biases


class Simple_NN(nn.Module):
    def __init__(self, input_size, output_size):
        super(Simple_NN, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, output_size)
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

        # Note: Softmax is not needed for cost calculation
        # Only apply softmax when doing inferencing
