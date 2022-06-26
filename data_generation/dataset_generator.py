# Imports
import numpy as np
from sklearn.datasets import make_classification
from keras.utils import to_categorical
from os.path import exists


# Functions
def generate_classification_dataset(n_samples, n_features, n_informative, n_redundant, n_classes, weights, target_dir="./dataset_1/"):
    """Generate classification problem dataset using sklearn"""
    features_exists = exists(target_dir + "data_features.csv")
    target_exists = exists(target_dir + "data_targets.csv")

    if (features_exists and target_exists):
        print(f"dataset_generator: files already exists at {target_dir}")
        return

    print(f"Generating {n_samples} samples")
    print(f"Input Size: {n_features}, Output Size: {n_classes}")

    features, target = make_classification(n_samples=n_samples,
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

    with open(target_dir + "data_meta.txt", "w") as f:
        f.write("n_samples, n_features, n_informative, n_redundant, n_classes, weights")
        f.write("\n")
        f.write(f"{n_samples}, {n_features}, {n_informative}, {n_redundant}, {n_classes}, {weights}")


# Main Function
if __name__ == "__main__":
    generate_classification_dataset(n_features=3,
                     n_informative=3,
                     n_redundant=0,
                     n_classes=2,
                     weights=[0.33, 0.67])
