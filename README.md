# training-NN-with-Graph-Learning
Project for CUHK Summer Research Internship

## Implementation
The core GNN will be implemented using the PyTorch Geometric library.

I've tried using Spektral, which is based on Keras/Tensorflow, and some legacy code is left behind.

## Training Data Structure
Since Training Data is a collection of graphs, the data is more structured than the usual ML training data.

There should be a `raw` folder, and inside of the raw folder, there are folders for models trained using different datasets.

```
.
├── raw
|   ├── dataset_1
|   |   ├── data_features.csv
|   |   ├── data_targets.csv
|   |   ├── model_designs.txt
|   |   ├── model_weights.txt
|   |   └── model_biases.txt
|   ├── dataset_2
|   └── ...
└── ...
```

The pytorch program will also create a `processed` folder for storing the processed graph data. This folder should be ignored by git.

## Purpose of each File

These files are related to training data generation:

- `raw/dataset_generator.py`: Creates a random dataset using scikit-learn's make-classification function
- `raw/model_generator.py`: Generates a random and trained dense NN model based on a dataset
- `raw/main.py`: Using the two python scripts, generate multiple datasets with the training data and trained NNs

These files are related to training the GNN:

- `dataset_graphs.py`: Converts the datasets into the dataset object in Torch Geometric/Spektral.
- `main.py`: Contains GNN model and training code