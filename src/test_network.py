import numpy as np
import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
import network


# Create network: 2 inputs, 3 hidden neurons, 1 output
net = network.Network([784, 30, 10])

# Train
# ETA is the learning rate
net.SGD(training_data, epochs=30, mini_batch_size=10, eta=3.0, test_data=test_data)

# Epoch 29: 9698 / 10000, took 16.28 seconds
# run in the src directory with "python3 test_network.py"