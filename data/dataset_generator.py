# Imports
import numpy as np
from sklearn.datasets import make_classification
from keras.utils import to_categorical


# Constants
SAMPLES = 500


# Functions
def generate_dataset(n_features, n_informative, n_redundant, n_classes, weights, target_dir="./dataset_1/"):
    print(f"Generating {SAMPLES} samples")
    print(f"Input Size: {n_features}, Output Size: {n_classes}")

    features, target = make_classification(n_samples=SAMPLES,
                                           n_features=n_features,
                                           n_informative=n_informative,
                                           n_redundant=n_redundant,
                                           n_classes=n_classes,
                                           weights=weights)

    # Converts target array to one hot encoding
    target = to_categorical(target)

    print(f"Saving to {target_dir}data_features.csv")
    np.savetxt(target_dir + "data_features.csv", features, delimiter=", ")

    print(f"Saving to {target_dir}data_targets.csv")
    np.savetxt(target_dir + "data_targets.csv", target, delimiter=", ")


# Main Function
if __name__ == "__main__":
    generate_dataset(n_features=3,
                     n_informative=3,
                     n_redundant=0,
                     n_classes=2,
                     weights=[0.33, 0.67])
