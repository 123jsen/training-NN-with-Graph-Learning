# Imports
import numpy as np
from sklearn.datasets import make_classification


# Constants
TARGET_DIR = "./data/dataset_1/"


# Main Function
if __name__ == "__main__":
    features, target = make_classification(n_samples=500,
                                           n_features=3,
                                           n_informative=3,
                                           n_redundant=0,
                                           n_classes=2,
                                           weights=[0.33, 0.67])

    np.savetxt(TARGET_DIR + "data_features.csv", features, delimiter=", ")
    np.savetxt(TARGET_DIR + "data_targets.csv", target, delimiter=", ")
