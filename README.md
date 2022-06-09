# training-NN-with-Graph-Learning
Project for CUHK Summer Research Internship

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
````
