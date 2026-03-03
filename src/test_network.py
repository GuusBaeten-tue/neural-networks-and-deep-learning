import numpy as np
import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
import network


# Create network: 784 inputs, 30 hidden neurons, 10 outputs
net = network.Network([784, 50, 10])

# Train
# ETA is the learning rate
net.SGD(training_data, epochs=30, mini_batch_size=10, eta=3.0, test_data=test_data)

# Epoch 29: 9698 / 10000
# run in the src directory with "python3 test_network.py"