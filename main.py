import model_generator

model = model_generator.rand_model(input_shape=(784,), num_output=10)

model.summary()