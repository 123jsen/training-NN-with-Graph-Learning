# training-NN-with-Graph-Learning

Project for CUHK Summer Research Internship. The goal is to train a Graph Neural Network that can train dense Neural Networks.

---

## Implementation

The core GNN will be implemented using the PyTorch Geometric library.

I've tried using Spektral, which is based on Keras/Tensorflow, and all legacy code is deleted, but can be found in the git logs.

### Documentation

For more details on the data and graph implementation, see `doc/data_format.md` and `doc/graph_structure.md`

---

## Workflow

1. Dataset Creation: Run `data_generation/main.py`, which first runs `dataset_generator.py` to generate classification data, then runs `model_generator.py` to generate dense neural networks that is trained on the classification data.

2. Graph Data Processing: Load the `NNDataset` from `datasets.graph_data` to process the text data into Pytorch Geometric's graph object as a dataset.

3. Training: Train a GNN.

4. Evaluation: Predict weights and biases using the GNN, then convert the numerical predictions into a Pytorch Dense Neural Network and evaluate the accuracy of the model.

### Utilities

The utility module helps with the workflow above. Step 1 converts models to text (`utility.model2text`), step 2 converts text to graphs (`utility.text2graph`), step 3 converts graphs back to models (`utility.graph2model`).
