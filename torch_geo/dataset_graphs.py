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

    def process(self):
        if (EDGES_DIRECTED):
            print("Graph is Directed")
        else:
            print("Graph is Undirected")

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
                    data.design = design

                    # Node input info
                    data.num_nodes = sum(design)

                    # TODO: data.x

                    # Sparse adjacency matrix
                    data.edge_index = torch.zeros((2, 0))

                    lb = 0      # lower bound
                    for index, height in enumerate(design[:-1]):
                        for i in range(height):
                            for j in range(height, height + design[index + 1]):
                                new_col = torch.tensor([[lb + i], [lb + j]])
                                if not(EDGES_DIRECTED):
                                    new_col = torch.cat(
                                        (new_col, torch.tensor([[lb + j], [lb + i]])), axis=1)

                                data.edge_index = torch.cat(
                                    (data.edge_index, new_col), axis=1)

                        lb += height

                    # Edge y
                    data.y_edge = torch.zeros(0)
                    for index, height in enumerate(design[:-1]):
                        for i in range(height):
                            weights = weights_file.readline()
                            weights = np.fromstring(weights, dtype=float, sep=', ')
                            data.y_edge = torch.cat(
                                (data.y_edge, torch.tensor(weights)))

                    # Node y
                    data.y_node = torch.zeros(0)
                    for index, height in enumerate(design):
                        # First rows do not have biases -> all zero
                        if (index == 0):
                            data.y_node = torch.cat(
                                (data.y_node, torch.zeros(height)))
                            continue

                        # Read biases values line by line
                        biases = biases_file.readline()
                        biases = np.fromstring(biases, dtype=float, sep=', ')
                        biases = torch.from_numpy(biases)
                        data.y_node = torch.cat((data.y_node, biases))

                    data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

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
