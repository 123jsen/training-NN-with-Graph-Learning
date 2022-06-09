### Imports ###
import random
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.optimizers import Adam
from keras.losses import CategoricalCrossentropy


### Constants ###
MAX_DEPTH = 5
LAYER_HEIGHTS = (4, 8, 16, 32, 64)


### Functions ###
def rand_design():
    '''Returns a tuple representing layers height'''
    depth = random.randrange(MAX_DEPTH) + 1
    result = np.asarray([random.choice(LAYER_HEIGHTS) for i in range(depth)])
    return result


def create_model(model_design, input_shape=(32,), num_output=2):
    '''Creates dense NN model using given parameters'''
    model = Sequential()

    model.add(Input(shape=input_shape))

    for depth in model_design:
        model.add(Dense(depth, activation="relu"))

    model.add(Dense(num_output, activation="relu"))

    return model


def rand_model(input_shape=(32,), num_output=2):
    design = rand_design()
    return design, create_model(design, input_shape, num_output)


def prepare_model_data(num_models=10, target_dir="./dataset_1/"):
    '''Read the data from the csv files, trains a bunch of NNs, and store them at the folder'''
    features = np.genfromtxt(target_dir + 'data_features.csv', delimiter=', ')
    targets = np.genfromtxt(target_dir + 'data_targets.csv', delimiter=', ')

    input_size = features[0].shape[0]
    output_size = targets.shape[1]

    print(f"Input Size = {input_size}, Output Size = {output_size}")

    for i in range(num_models):
        print(f"Generating the {i}-th model")
        # This part trains the model
        design, model = rand_model(input_shape=(input_size, ),
                                   num_output=output_size)

        # model.summary()

        design = np.concatenate(([input_size], design, [output_size]))
        print("\nDesign =", design)

        model.compile(optimizer=Adam(learning_rate=0.01),
                      loss=CategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        print("Training...")
        history = model.fit(features, targets, epochs=100, verbose=0)

        print("loss =", history.history['loss'][-1])
        print("acc =", history.history['accuracy'][-1])

        # This part appends the model to text file
        # Design
        with open(target_dir + 'model_designs.txt', 'a') as designs_file:
            np.savetxt(designs_file, design, fmt="%d, ", newline="")
            designs_file.write("\n")


        for layer in model.layers:
            weights = layer.get_weights()
            # Weights
            with open(target_dir + 'model_weights.txt', 'a') as weights_file:
                np.savetxt(weights_file, weights[0], delimiter=", ")

            # Biases
            with open(target_dir + 'model_biases.txt', 'a') as biases_file:
                np.savetxt(biases_file, weights[1], newline=", ")
                biases_file.write("\n")

        print("Model data written to file")


### Main Function ###
if __name__ == "__main__":
    prepare_model_data(num_models=10)
