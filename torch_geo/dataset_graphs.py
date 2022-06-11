### Imports ###
import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset


### Constants ###
EDGES_DIRECTED = False
SOURCES = ["dataset_1/", "dataset_2/", "dataset_3/"]


class NNDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return SOURCES

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        print("Data download not implemented")
        print("Do you have the correct files at /data folder?")
        exit(1)

    def read_designs(self, path):
        """Reads array of designs from given path"""

        designs = []
        with open(path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                design = []

                line = line.rstrip("\n")
                line = line.split(', ')

                for num in line:
                    try:
                        if num:
                            design.append(int(num))
                    except ValueError:
                        print(f"Warning: Cannot append [{num}]")

                designs.append(design)

        return designs

    def process(self):

        data_list = []

        for path in self.raw_paths:
            # One dataset folder represents multiple NN trained on the same data
            print(f"Reading from {path}", end=", ")

            features = np.genfromtxt(
                path + "data_features.csv", delimiter=', ')

            targets = np.genfromtxt(path + "data_targets.csv", delimiter=',')

            designs = self.read_designs(path + "model_designs.txt")
            print(f"{len(designs)} models loaded")

            with open(path + "model_weights.txt", 'r') as weights_file, open(path + "model_biases.txt", 'r') as biases_file:
                for design in designs:
                    # One design is one neural network graph
                    data = Data()
                    data.num_nodes = sum(design)

                    data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
