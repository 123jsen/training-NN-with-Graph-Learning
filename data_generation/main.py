# Imports
import os
from model_generator import generate_model_data
from dataset_generator import generate_classification_dataset


# Constants
from constants.constants import SAMPLES_PER_SET, MODELS_PER_SET

# Data parameters
features = [3, 4, 6, 7, 4, 9, 8, 5, 4]

informative = [3, 2, 5, 5, 4, 7, 5, 5, 3]

redundant = [0, 2, 1, 2, 0, 2, 3, 0, 1]

classes = [2, 2, 3, 3, 2, 4, 2, 3, 2]

weights = [[0.33, 0.67],
           [0.5, 0.5],
           [0.35, 0.25, 0.4],
           [0.1, 0.1, 0.8],
           [0.4, 0.6],
           [0.25, 0.25, 0.25, 0.25],
           [0.63, 0.37],
           [0.3, 0.3, 0.4],
           [0.7, 0.3]]


# Main Function
if __name__ == "__main__":
    '''This program creates a lot of datasets and models for the training of the GNN'''

    for i in range(len(features)):
        # Create folder if it doesnt exists
        path = f"./raw/dataset_{i}/"
        os.makedirs(path, exist_ok=True)

        generate_classification_dataset(n_samples=SAMPLES_PER_SET,
                                        n_features=features[i],
                                        n_informative=informative[i],
                                        n_redundant=redundant[i],
                                        n_classes=classes[i],
                                        weights=weights[i],
                                        target_dir=path)

        generate_model_data(num_models=MODELS_PER_SET,
                            target_dir=path)
