import torch.nn as nn
import torch.nn.functional as F

class DenseNet(nn.Module):
    def __init__(self, design):
        '''Creates dense NN given a particular design'''
        super(DenseNet, self).__init__()
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