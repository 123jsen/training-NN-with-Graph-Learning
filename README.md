# training-NN-with-Graph-Learning
Project for CUHK Summer Research Internship

## Implementation
The core GNN will be implemented using Spektral, which is a graph learning API based on Keras/Tensorflow.

## Training Data Structure
Since Training Data is a collection of graphs, the data is more structured than the usual ML training data.

There should be a `data` folder, and inside of the data folder, there are folders for models trained using different datasets.

```
.
├── data
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

## Purpose of each File

- `data/dataset_generator.py`: Creates a random dataset using scikit-learn's make-classification function
- `data/model_generator.py`: Generates a random and trained dense NN model based on a dataset
- `data/main.py`: Using the two python scripts, generate multiple datasets with the training data and trained NNs

- `dataset_graphs.py`: Converts the datasets into the dataset object in Spektral API