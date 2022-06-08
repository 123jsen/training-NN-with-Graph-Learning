# training-NN-with-Graph-Learning
Project for CUHK Summer Research Internship

## Training Data Structure
Since Training Data is a collection of graphs, the data is more structured than the usual ML training data.

There should be a `data` folder, and inside of the data folder, there are folders for models trained using different datasets.

```
.
├── data
|   ├── dataset_1
|   |   ├──train_data
|   |   ├──model_design
|   |   ├──model_weights
|   |   └──model_biases
|   ├── dataset_2
|   └── ...
└── ...
````
