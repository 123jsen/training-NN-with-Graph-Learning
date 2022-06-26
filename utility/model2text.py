import numpy as np
import os


def write_design(filename, design):
    """Write design array at target directory"""
    # Design
    with open(filename, 'a') as designs_file:
        np.savetxt(designs_file, design, fmt="%d, ", newline="")
        designs_file.write("\n")


def write_weights(filename_weights, filename_biases, model):
    """Write model weights and biases are target directory"""
    # Weights and Biases
    for i in range(len(model.layers)):
        weights = model.state_dict()[f"layers.{i}.weight"].cpu()
        with open(filename_weights, 'a') as weights_file:
            np.savetxt(weights_file, weights.T, delimiter=", ")

        biases = model.state_dict()[f"layers.{i}.bias"].cpu()
        with open(filename_biases, 'a') as biases_file:
            np.savetxt(biases_file, biases.T, newline=", ")
            biases_file.write("\n")


def write_metrics(filename, loss, acc):
    """Write training_metrics at target_dir"""
    # Metrics
    if not(os.path.isfile(filename + 'model_metrics.txt')):
        with open(filename + 'model_metrics.txt', 'w') as meta_file:
            meta_file.write("loss, accuracy")
            meta_file.write("\n")

    with open(filename + 'model_metrics.txt', 'a') as meta_file:
        meta_file.write(f"{loss}, {acc}\n")

