from dataset_graphs import NNDataset

if __name__ == "__main__":
    dataset = NNDataset(root="")
    
    for i in [1]:
        print(f"graph {i}-th design: {dataset[i].design}")
        print(f"graph {i}-th num nodes: {dataset[i].num_nodes}")
        print(f"graph {i}-th num edges: {dataset[i].num_edges}")
        print(dataset[i].edge_index)