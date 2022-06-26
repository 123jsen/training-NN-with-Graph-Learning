# Structure of Graph Data
In our project, every neural network is a graph. The graph is implemented using PyTorch Geometric's graph class. The graph is loaded from the datasets and processed in the `graph_data.py` file.

This document explains every attribute of the NN graph. See the [Pytorch Geometric Documentation](https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html#data-handling-of-graphs) for unmentioned details.

---

## Attributes

### Node Related

- `graph.x`: Node input feature matrix. Has shape `[num_nodes, SAMPLES_PER_SET + 3]`. First `SAMPLES_PER_SET` contains input, output or zeros data for the node. The next three represents `(is_input, is_output, initial_bias)` 

- `graph.node_y`: Node output biases matrix. Has shape `[num_nodes, 1]`.

- `graph.node_mask`: Mask to decide which nodes to compare and which not to. Multiply the output of the GNN model with the mask pointwisely to get zeroes for input layer outputs. Has shape `[num_nodes, 1]`

### Edge Related

- `graph.edge_index`: Connectivity Matrix. Has shape `[2, num_edges]`

- `graph.edge_attr`: Input edge weights. Has shape `[num_edges, 1]`

- `graph.edge_y`: Output edge weights. Has shape `[num_edges, 1]`

### Other Attributes

- `graph.design`: Array containing number of nodes per layer (including input and output).