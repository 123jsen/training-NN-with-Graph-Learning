# Imports
from model_generator import prepare_model_data
from dataset_generator import generate_dataset


# Constants
SAMPLES = 500
MAX_FEATURE_NUM = 10
MAX_OUTPUT_NUM = 10
MODELS_PER_SET = 50


# Main Function
if __name__ == "__main__":
    '''This program creates a lot of datasets and models for the training of the GNN'''

    generate_dataset(n_features=3,
                     n_informative=3,
                     n_redundant=0,
                     n_classes=2,
                     weights=[0.33, 0.67],
                     target_dir="./data/dataset_1/")
    prepare_model_data(num_models=MODELS_PER_SET, target_dir="./data/dataset_1/")

    generate_dataset(n_features=4,
                     n_informative=2,
                     n_redundant=2,
                     n_classes=2,
                     weights=[0.5, 0.5],
                     target_dir="./data/dataset_2/")
    prepare_model_data(num_models=MODELS_PER_SET, target_dir="./data/dataset_2/")

    generate_dataset(n_features=6,
                     n_informative=5,
                     n_redundant=1,
                     n_classes=3,
                     weights=[0.35, 0.25, 0.4],
                     target_dir="./data/dataset_3/")
    prepare_model_data(num_models=MODELS_PER_SET, target_dir="./data/dataset_3/")
