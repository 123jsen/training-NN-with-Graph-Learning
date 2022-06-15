# Downloaded libraries
import torch
from torch import nn
from torch_geometric.loader import DataLoader


# Local files
from dataset_graphs import NNDataset
from models import Trainer_GCN


# Constants
TRAINING_SPLIT = 0.8


# Hyperparameters
num_epoch = 50
batch_size = 16

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    nndataset = NNDataset(root="")

    size = len(nndataset)
    train_num = int(size * TRAINING_SPLIT)
    test_num = size - train_num

    train_loader = DataLoader(dataset=nndataset[:train_num], batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=nndataset[train_num:], batch_size=test_num, shuffle=True)

    print(f"Dataset loaded, {train_num} training samples and {test_num} testing samples")

    model = Trainer_GCN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    loss_fn = nn.MSELoss()

    # Model Training
    model.train()
    for epoch in range(num_epoch):
        print(f"Epoch {epoch + 1} / {num_epoch}:")
        for i, data in enumerate(train_loader):
            data.to(device)

            # forward propagation
            out_w, out_b = model(data)
            loss = loss_fn(out_b, data.y_node) + loss_fn(out_w, data.y_edge)

            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print status every n batches
            if i % 10 == 0:
                loss, current = loss.item(), i * batch_size
                print(f"Training Loss: {loss:>7f}  [{current:>5d}/{train_num:>5d}]")
    print("Training complete")

    # Model Evaluation
    model.eval()
    with torch.no_grad():
        data = iter(test_loader).next().to(device)

        # forward propagation
        out_w, out_b = model(data)
        loss = loss_fn(out_b, data.y_node) + loss_fn(out_w, data.y_edge)

        loss = loss.item()
        print(f"Validation Loss: {loss:>7f}")

    torch.save(model.state_dict(), "./model/model")
    print("Model saved")
