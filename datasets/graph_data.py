### Imports ###
import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset


### Constants ###
EDGES_DIRECTED = True       # The Graph can be changed to undirected using PyG transforms
SOURCES = ["dataset_0/", "dataset_1/", "dataset_2/",
           "dataset_3/", "dataset_4/", "dataset_5/",
           "dataset_6/", "dataset_7/", "dataset_8/"]

from constants.constants import SAMPLES_PER_SET


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
        print("Do you have the correct files at /raw folder?")
        raise Exception("Dataset not configured correctly")

    def process(self):

        data_list = []

        for path in self.raw_paths:
            # One dataset folder represents multiple NN trained on the same data
            print(f"Reading from {path}", end=", ")

            features = np.genfromtxt(
                path + "data_features.csv", delimiter=', ')

            targets = np.genfromtxt(path + "data_targets.csv", delimiter=',')

            designs = read_designs(path + "model_designs.txt")

            with open(path + "model_weights.txt", 'r') as weights_file, \
                    open(path + "model_biases.txt", 'r') as biases_file, \
                    open(path + "model_init_biases.txt", 'r') as init_biases_file, \
                    open(path + "model_init_weights.txt", 'r') as init_weights_file:

                for i, design in enumerate(designs):
                    # One design is one neural network graph
                    graph = graph_from_design(design)

                    # Node x
                    fill_node_x(graph, design, init_biases_file,
                                features, targets)

                    # Edge x
                    fill_edge_x(graph, design, init_weights_file)

                    # Node y
                    fill_node_y(graph, design, biases_file)

                    # Edge y
                    fill_edge_y(graph, design, weights_file)

                    # Mask for comparing Node y
                    graph.node_mask = mask_from_graph(graph)

                    data_list.append(graph)

            print(f"{len(designs)} models loaded")

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def read_designs(path):
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


def graph_from_design(design):
    '''Produces graph data object from a given design array'''

    graph = Data()
    graph.design = design

    # Node input info
    graph.num_nodes = sum(design)

    # Sparse adjacency matrix
    graph.edge_index = torch.zeros((2, 0), dtype=torch.long)

    lb = 0      # lower bound
    for index, height in enumerate(design[:-1]):
        for i in range(height):
            for j in range(height, height + design[index + 1]):
                new_col = torch.tensor(
                    [[lb + i], [lb + j]], dtype=torch.long)

                graph.edge_index = torch.cat(
                    (graph.edge_index, new_col), axis=1)

        lb += height

    return graph


def fill_node_x(graph, design, file, features, targets):
    '''Configure graph.x to contain input data'''

    # Data x, beware this error https://stackoverflow.com/questions/67481937/indexerror-dimension-out-of-range-expected-to-be-in-range-of-1-0-but-got

    # First NUM_SAMPLES encode training data information (input, output, or none)
    # Next 3 encodes: is_input, is_output, initial_bias

    graph.x = np.zeros((graph.num_nodes, SAMPLES_PER_SET))

    graph.x[:design[0]] = features.T        # input layer
    graph.x[-design[-1]:] = targets.T        # output layer

    # np.pad directions: [(Up, Down), (Left, Right)]
    graph.x = np.pad(graph.x, [(0, 0), (0, 3)])

    graph.x[:design[0], SAMPLES_PER_SET + 0] = 1
    graph.x[-design[-1]:, SAMPLES_PER_SET + 1] = 1

    # load initial biases from file
    graph.x[:design[0], SAMPLES_PER_SET + 2] = 0

    graph.x = torch.tensor(graph.x, dtype=torch.float32)

    init_biases = torch.zeros(0, dtype=torch.float32)
    for i in range(len(design[1:])):
        bias = file.readline()
        bias = np.fromstring(
            bias, dtype=np.float32, sep=', ')
        bias = torch.from_numpy(bias)
        init_biases = torch.cat((init_biases, bias))

    graph.x[design[0]:, SAMPLES_PER_SET + 2] = init_biases


def fill_edge_x(graph, design, file):
    graph.edge_attr = torch.zeros(0)
    for index, height in enumerate(design[:-1]):
        for j in range(height):
            weights = file.readline()
            weights = np.fromstring(
                weights, dtype=np.float32, sep=', ')
            graph.edge_attr = torch.cat(
                (graph.edge_attr, torch.tensor(weights)))

    graph.edge_attr = graph.edge_attr.reshape([-1, 1])


def fill_node_y(graph, design, file):
    graph.node_y = torch.zeros(0, dtype=torch.float32)
    for index, height in enumerate(design):
        # First rows do not have biases -> all zero
        if (index == 0):
            graph.node_y = torch.cat(
                (graph.node_y, torch.zeros(height, dtype=torch.float32)))
            continue

        # Read biases values line by line
        biases = file.readline()
        biases = np.fromstring(
            biases, dtype=np.float32, sep=', ')
        biases = torch.from_numpy(biases)
        graph.node_y = torch.cat((graph.node_y, biases))

    graph.node_y = graph.node_y.reshape([-1, 1])


def fill_edge_y(graph, design, file):
    graph.edge_y = torch.zeros(0)
    for index, height in enumerate(design[:-1]):
        for j in range(height):
            weights = file.readline()
            weights = np.fromstring(
                weights, dtype=np.float32, sep=', ')
            graph.edge_y = torch.cat(
                (graph.edge_y, torch.tensor(weights)))

    graph.edge_y = graph.edge_y.reshape([-1, 1])


def mask_from_graph(data):
    mask = torch.ones((data.num_nodes, 1), dtype=int)
    mask[:data.design[0]] = torch.zeros((data.design[0], 1), dtype=torch.int)

    return mask


if __name__ == "__main__":
    NNDataset(root="")
