import numpy as np
import torch
from torch.utils.data import Dataset

class ClassificationData(Dataset):
    def __init__(self, dir, transform=None, target_transform=None):
        self.features = np.genfromtxt(dir + "data_features.csv", delimiter = ", ")
        self.features = torch.tensor(self.features, dtype=torch.float32)
        
        self.targets =  np.genfromtxt(dir + "data_targets.csv", delimiter = ", ")
        self.targets = torch.tensor(self.targets, dtype=torch.float32)

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        target = self.targets[idx]

        if self.transform:
            feature = self.transform(feature)
        if self.target_transform:
            target = self.target_transform(target)

        return feature, target