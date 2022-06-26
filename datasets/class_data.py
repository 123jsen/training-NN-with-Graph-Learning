import numpy as np
import torch
from torch.utils.data import Dataset

class ClassificationDataset(Dataset):
    '''PyTorch dataset object for classification training data'''
    def __init__(self, dest_dir):
        self.features = np.genfromtxt(
            dest_dir + 'data_features.csv', delimiter=', ')
        self.features = torch.tensor(self.features, dtype=torch.float32)

        self.targets = np.genfromtxt(
            dest_dir + 'data_targets.csv', delimiter=', ')
        self.targets = torch.tensor(self.targets, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.targets[index]