"""
    Created by Justin Weiss on 3/1/20.
    Copyright © 2020 Justin Weiss. All rights reserved.

    Feed Forward Multilayer Neural Network

    This program implements a feed forward neural network with no back propagation. The weights and the bias's are set
        below and never change. It takes in a csv file with examples and outputs a value between 0 and 1. It creates a
        neural network with three layers the input layer with 10 input nodes, the hidden layer with 10 nodes and the
        output layer with 1 node.
    NOTE: This program requires pandas and numpy, without these libraries the program will not run.
    RUN: python3 NeuralNetwork.py
"""

import pandas as pd
import numpy as np

# Global variables for the network array
HIDDEN = 0
OUTPUT = 1

# Layer class that holds each layers weights and bias
class Layer (object):

    def __init__(self, weights, bias=1):
        super(Layer, self).__init__()
        self.weights = weights
        self.bias = bias


# Initializes the network, it takes in the hidden layer bias and the output layer bias. It then creates a layer class
# element and adds it an list. It return a list making up the networks layers
def initialize_network(biasHidden, biasOutput):
    network = []

    # Hidden layer weights 10X10
    hiddenWeights = [[0.116, 2.136, 2.8, 1.45, 1.36, 0.0625, -1.25, 0.9937, -0.3286, -2.142],
                     [-0.34, 2.33, 1.23, 1.91, 1.53, -3.14, -0.0799, 0.19, -0.915, -1.64],
                     [1.74, -0.405, 0.55, -2.12, -2.434, 1.11, 0.28, -0.63, 1.01, 3.736],
                     [0.57, 0.1169, 0.27, 1.094, 1.14, 0.69, -0.69, 1.07, 0.012, -0.035],
                     [0.73, 0.462, 0.94, 0.47, 0.977, 0.74, -0.035, 0.6427, 0.1168, 0.821],
                     [0.55, -0.09199, 0.2269, 0.78, 0.347, 0.14, 0.429, 0.69, 0.432, 0.28],
                     [0.55, 0.652, 0.8, -0.832, -0.33, -0.96, -0.511, -0.337, 1.07, 1.91],
                     [0.16, 0.45, 0.98, 0.76, 0.526, 0.856, 0.178, 0.512, 0.85, 0.55],
                     [-3.089, -3.99, 3.809, -1.48, -0.1924, 0.175, 1.919, 0.07, 0.70, 0.12],
                     [3.160, 4.04, -3.72, 0.73, 0.74, 0.108, -2.14, -0.03, -0.79, 0.86]]

    # Output layer weights 1X10
    outputWeights = [[-0.1, -1, 2.55, 3.12, 0, -1, -2.39, -2, 1.5, -0.5]]

    # Creates both layers and adds them to the list
    hidden = Layer(hiddenWeights, biasHidden)
    network.append(hidden)
    output = Layer(outputWeights, biasOutput)
    network.append(output)

    # returns the list of layers
    return network


# This function takes in a value and return the sigmoid function, a value between 0 and 1
def sigmoid(value):
    return 1 / (1 + np.exp(-value))


# This function takes in the weights from the neuron, the input data, and the layer bias. It adds the bias and multiples
# the weights with the input data, and returns the sum
def activate(weight, inputData, bias):
    sum = bias

    # Loops through the values of weight
    for x in range(len(weight)):
        sum += weight[x] * inputData[x]
    return sum

# Forward propagates input to a network output, it takes in the network, and the input data. It loops through all the
# layers, and weights calculating the activition function and running that data through the sigmoid function then
# return in the value for output
def forwardPropagate(network, input):
    for layer in network:
        newInputs = []
        for weight in layer.weights:
            output = activate(weight, input, layer.bias)
            output = sigmoid(output)
            newInputs.append(output)
        input = newInputs
    return input


# This function cleans the data, and changes the values from strings to ints
def cleanData (x):
    return x.replace("Yes", 1).replace("No", 0).replace("None", 0).replace("Some", 1).replace("Full", 2).\
        replace("$", 0).replace("$$", 1).replace("$$$", 2).replace("French", 0).replace("Thai", 1).\
        replace("Burger", 2).replace("Italian", 3).replace("0-10", 0).replace("10-30", 1).replace("30-60", 2). \
        replace(">60", 3)


def main():
    # Read in data and clean whitespace char
    dataFile = pd.read_csv('restaurant.csv', header=None)
    dataFile = dataFile.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

    # applys the clean data function to every value in the input data set
    dataFile = dataFile.apply(cleanData)

    # deletes the will wait column, creates a list from the input data, and creates the network
    del dataFile[10]
    inputs = dataFile.values.tolist()
    network = initialize_network(1, 1)

    # Loops through all the input data examples, call forward propagate on all and prints the values to the screen
    count = 1
    for input in inputs:
        output = forwardPropagate(network, input)
        print("Example %2.d: %.10f" %(count, float(output[0])))
        count += 1


main()
