# Packages Imports
import numpy as np
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Dense
from spektral.layers import GCNConv
from spektral.data import BatchLoader
from spektral.transforms import GCNFilter


# Custom Imports
from dataset_graphs import NNDataset


class TrainerGNN(Model):

    def __init__(self, n_hidden):
        super().__init__()
        self.graph_conv = GCNConv(n_hidden)
        self.dense = Dense(1)                   # For regression

    def call(self, inputs):
        print("\nHello\n")
        out = self.graph_conv(inputs)
        print("\nBye bye\n")
        out = self.dense(out)

        return out

if __name__ == "__main__":
    dataset = NNDataset()
    dataset.apply(GCNFilter())

    loader = BatchLoader(dataset, batch_size=4)

    model = TrainerGNN(n_hidden=64)
    model.compile(optimizer=Adam(learning_rate=0.01),
                  loss='mean_squared_error')

    model.fit(loader.load(), steps_per_epoch=loader.steps_per_epoch, epochs=10)
