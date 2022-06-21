# Imports
import os
from model_generator import generate_model_data
from dataset_generator import generate_classification_dataset


# Constants
SAMPLES = 500
MAX_OUTPUT_NUM = 10
MODELS_PER_SET = 200
NUM_DATASETS = 6

# Data parameters
features = [3, 4, 6, 7, 4, 9]

informative = [3, 2, 5, 5, 4, 7]

redundant = [0, 2, 1, 2, 0, 2]

classes = [2, 2, 3, 3, 2, 4]

weights = [[0.33, 0.67],
           [0.5, 0.5],
           [0.35, 0.25, 0.4],
           [0.1, 0.1, 0.8],
           [0.4, 0.6],
           [0.25, 0.25, 0.25, 0.25]]


# Main Function
if __name__ == "__main__":
    '''This program creates a lot of datasets and models for the training of the GNN'''

    for i in range(5):
        # Create folder if it doesnt exists
        path = f"./raw/dataset_{i}/"
        os.makedirs(path, exist_ok=True)

        generate_classification_dataset(n_features=features[i],
                         n_informative=informative[i],
                         n_redundant=redundant[i],
                         n_classes=classes[i],
                         weights=weights[i],
                         target_dir=path)

        generate_model_data(num_models=MODELS_PER_SET,
                           target_dir=path)
