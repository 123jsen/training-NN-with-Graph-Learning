import random
import keras
from keras.models import Sequential
from keras.layers import Dense, Input

# Constants
MAX_DEPTH = 5
LAYER_HEIGHTS = (4, 8, 16, 32, 64)

# Functions

# Returns a tuple representing layers height
def rand_design():
    depth = random.randrange(MAX_DEPTH) + 1
    result = tuple([random.choice(LAYER_HEIGHTS) for i in range(depth)])
    return result

def create_model(model_design, input_shape=(32,), num_output=2):
    model = Sequential()

    model.add(Input(shape=input_shape))

    for depth in model_design:
        model.add(Dense(depth, activation="relu"))

    model.add(Dense(num_output, activation="relu"))

    return model

def rand_model(input_shape=(32,), num_output=2):
    design = rand_design()
    return create_model(design, input_shape, num_output)
