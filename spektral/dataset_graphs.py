### Imports ###
import numpy as np
from spektral.data import graph, dataset


### Constants ###
EDGES_DIRECTED = False
SOURCES = ["../data/dataset_1/", "../data/dataset_2/", "../data/dataset_3/"]


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


class NNDataset(dataset.Dataset):
    '''Dataset Object that returns NN graphs
       We will first experiment whether we can predict biases'''

    def read(self):
        print("Reading Data from text files")

        output = []

        for data_source in SOURCES:
            print()

            print(f"Reading from {data_source}")

            features = np.genfromtxt(
                data_source + "data_features.csv", delimiter=', ')
            print(f"Features: shape {features.shape}")

            targets = np.genfromtxt(
                data_source + "data_targets.csv", delimiter=',')
            print(f"Targets: shape {targets.shape}")

            designs = read_designs(data_source + "model_designs.txt")
            print(f"Number of models: {len(designs)}")

            with open(data_source + "model_weights.txt", 'r') as weights_file, open(data_source + "model_biases.txt", 'r') as biases_file:
                for design in designs:
                    num_nodes = sum(design)

                    # Adjacency Matrix - Assumes that all layers are Dense
                    a = np.zeros((num_nodes, num_nodes))

                    # lower bound
                    lb = 0
                    for index, height in enumerate(design):
                        if (index == len(design) - 1):
                            break

                        for i in range(height):
                            for j in range(height, height + design[index + 1]):
                                a[lb + i][lb + j] = 1

                        lb += height

                    if not(EDGES_DIRECTED):
                        a += a.T

                    # Node Features
                    # 500 data, 10 deep each
                    x = np.zeros((num_nodes, 5000))

                    # Edge Features

                    # Labels
                    y = np.zeros(0)
                    for index, height in enumerate(design):
                        # First rows do not have biases -> all zero
                        if (index == 0):
                            y = np.zeros((height, ))
                            continue

                        # Read biases values line by line
                        biases = biases_file.readline()
                        biases = np.fromstring(biases, dtype=float, sep=', ')
                        y = np.concatenate((y, biases))

                    output.append(
                        graph.Graph(a=a, x=x, y=y)
                    )

        return output


if __name__ == "__main__":
    '''Try to init the object to run read'''
    data = NNDataset()