### Imports ###
import numpy as np
import torch
from torch_geometric.data import InMemoryDataset


### Constants ###
EDGES_DIRECTED = False
SOURCES = ["../data/dataset_1/", "../data/dataset_2/", "../data/dataset_3/"]


class NNDataset()