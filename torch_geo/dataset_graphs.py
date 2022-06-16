### Imports ###
import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset


### Constants ###
EDGES_DIRECTED = False
SOURCES = ["dataset_0/", "dataset_1/",
           "dataset_2/", "dataset_3/", "dataset_4/"]
NUM_SAMPLES = 500


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
        raise Exception("Dataset not configured correctly")

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

            designs = read_designs(path + "model_designs.txt")
            print(f"{len(designs)} models loaded")

            with open(path + "model_weights.txt", 'r') as weights_file, open(path + "model_biases.txt", 'r') as biases_file:
                for i, design in enumerate(designs):
                    # One design is one neural network graph
                    data = graph_from_design(design)

                    # Node x
                    populate_node_x(data, design, features, targets)

                    # Edge y
                    data.y_edge = torch.zeros(0)
                    for index, height in enumerate(design[:-1]):
                        for i in range(height):
                            weights = weights_file.readline()
                            weights = np.fromstring(
                                weights, dtype=np.float32, sep=', ')
                            data.y_edge = torch.cat(
                                (data.y_edge, torch.tensor(weights)))

                    data.y_edge = data.y_edge.reshape([-1, 1])

                    # Node y
                    data.y_node = torch.zeros(0, dtype=torch.float32)
                    for index, height in enumerate(design):
                        # First rows do not have biases -> all zero
                        if (index == 0):
                            data.y_node = torch.cat(
                                (data.y_node, torch.zeros(height, dtype=torch.float32)))
                            continue

                        # Read biases values line by line
                        biases = biases_file.readline()
                        biases = np.fromstring(
                            biases, dtype=np.float32, sep=', ')
                        biases = torch.from_numpy(biases)
                        data.y_node = torch.cat((data.y_node, biases))

                    data.y_node = data.y_node.reshape([-1, 1])

                    # Mask for comparing Node y
                    data.input_mask = torch.ones((data.num_nodes, 1), dtype=int)
                    data.input_mask[:data.design[0]] = torch.zeros((data.design[0], 1), dtype=torch.int)

                    data_list.append(data)

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
                if not(EDGES_DIRECTED):
                    new_col = torch.cat(
                        (new_col, torch.tensor([[lb + j], [lb + i]], dtype=torch.long)), axis=1)

                graph.edge_index = torch.cat(
                    (graph.edge_index, new_col), axis=1)

        lb += height

    return graph


def populate_node_x(data, design, features, targets):
    '''Configure data.x to contain input data'''

    # Data x, beware this error https://stackoverflow.com/questions/67481937/indexerror-dimension-out-of-range-expected-to-be-in-range-of-1-0-but-got
    data.x = np.random.normal(size=(data.num_nodes, NUM_SAMPLES))

    data.x[0:design[0]] = features.T        # input layer
    data.x[-design[-1]:] = targets.T        # output layer

    data.x = torch.tensor(data.x, dtype=torch.float32)

if __name__ == "__main__":
    NNDataset(root="")