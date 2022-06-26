import torch
import torch.nn as nn

def assign_biases(model, bias_tensor, input_size):
    bias_tensor = bias_tensor[input_size:]

    with torch.no_grad():
        ptr = 0
        for layer in model.layers:
            layer_bias = bias_tensor[ptr:ptr + layer.bias.numel()]
            layer_bias = layer_bias.reshape(-1)
            layer.bias = nn.Parameter(layer_bias.T)
            ptr += len(layer.bias)


def assign_weights(model, weight_tensor):
    with torch.no_grad():
        ptr = 0
        for layer in model.layers:
            layer_weight = weight_tensor[ptr:ptr + layer.weight.numel()]
            layer_weight = layer_weight.reshape(layer.in_features, layer.out_features)
            layer.weight = nn.Parameter(layer_weight.T)
            ptr += len(layer.weight)