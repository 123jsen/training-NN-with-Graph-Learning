import gen_model

model = gen_model.rand_model(input_shape=(784,), num_output=10)

model.summary()