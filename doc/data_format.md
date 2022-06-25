# Format of Data
This document explains the structure, shape and details of the `/raw/dataset` files. Reading the files in the suggested order should be more sensible than the alphanumerical order.

### Notation

Denote the total number of sample data (of one dataset) as $N_D$. Denote the input length of the classification data as $n$, and the output length of the data as $m$.

Denote the total number of trained neural networks as $N_N$.

---

## Classification Training Dataset for Simple NN

1. `data_meta.txt` : First row is the header, second row contains information about the dataset which is used in sklearn's `make_classification`.

2. `data_features.csv` : Each row contains $n$ data, and there are $N_D$ rows in total.

3. `data_targets.csv` : Each row contains $m$ data, and there are $N_D$ rows in total.

---

## Data of Trained Neural Network

1. `model_designs.txt` : Each row contains a varying number of integers, where each integer represents the number of neurons on each layer (including input and output layers) There are $N_N$ rows in total.

2. `model_init_biases.txt` : Each row contains the initial biases of one layer in a neural network. The length of each row depends on the size of that layer, which can be found in the design file. The total number of rows is the total number of layers in all $N_N$ models. (Excluding the first, as the first layer has no bias)

3. `model_biases.txt` : The trained biases of the models. Structure is same as above.

4. `model_init_weights.txt` : Contains blocks of intial weights of connections between two layers (consider as a matrix). Each matrix has number of rows same as source layer and number of columns same as destination layer. The total number of such matrices depend on the design of the neural network, and all matrices of the $N_N$ neural networks are inside this file.

5. `model_weights.txt` : Same as above, but for weights after training.

6. `model_metrics.txt` : First row is the header. There are $N_N$ rows of corresponding training metric data.