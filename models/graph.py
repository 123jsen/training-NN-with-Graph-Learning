from torch import nn
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class TrainerGCN(nn.Module):
    '''Main focus of the project: GCN that trains NN'''
    def __init__(self):
        super().__init__()
        self.conv1 = GATConv(in_channels=503, out_channels=128, edge_dim=1)
        self.conv2 = GATConv(in_channels=128, out_channels=128, edge_dim=1)

        self.dense_1B = Linear(128, 1)
        # self.dense_2B = Linear(32, 1)


        self.dense_1W = Linear(128, 1)
        # self.dense_2W = Linear(32, 1)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_weight

        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)

        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)
        

        # Linear layer for biases pred
        b = self.dense_1B(x)
        # b = self.dense_2B(b)
        # Set all biases on input layer to zero
        b *= data.input_mask

        # Edge contains sum of adj nodes: https://github.com/pyg-team/pytorch_geometric/discussions/3554
        src, dst = data.edge_index
        if not(data.is_directed()):
            src, dst = src[::2], dst[::2]       # skip repeated edges for undirected graph
        w = (x[src] + x[dst]) / 2
        w = self.dense_1W(w)
        # w = self.dense_2W(w)

        return w, b
