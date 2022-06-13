# Downloaded libraries
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.loader import DataLoader


# Local files
from dataset_graphs import NNDataset


# Hyperparameters
num_epoch = 15
batch_size = 16


class Bias_PredGCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(1, 32)
        self.conv2 = GCNConv(32, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return x

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    nndataset = NNDataset(root="")

    size = len(nndataset)
    train_num = int(size * 0.8)
    test_num = size - train_num

    train_loader = DataLoader(dataset=nndataset[:train_num], batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=nndataset[train_num:], batch_size=test_num, shuffle=True)

    print(f"Dataset loaded, {train_num} training samples and {test_num} testing samples")

    model = Bias_PredGCN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    loss_fn = nn.MSELoss()

    # Model Training
    model.train()
    for epoch in range(num_epoch):
        print(f"Epoch {epoch + 1} / {num_epoch}:")
        for i, data in enumerate(train_loader):
            data.to(device)

            # forward propagation
            out = model(data)
            loss = loss_fn(out, data.y_node)

            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print status every 5 batches
            if i % 10 == 0:
                loss, current = loss.item(), i * batch_size
                print(f"Training Loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


    # Model Evaluation
    model.eval()
    with torch.no_grad():
        data = iter(test_loader).next().to(device)

        # forward propagation
        out = model(data)
        loss = loss_fn(out, data.y_node)

        loss = loss.item()
        print(f"Validation Loss: {loss:>7f}")
