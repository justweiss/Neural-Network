# Neural Network

The figure below shows a simple mathematical model of the neuron devised by McCulloch and Pitts. Roughly speaking, it “fires” when a linear combination of its inputs exceeds some (hard or soft) threshold that is, it implements a linear classifier of the kind described in the preceding section. A neural network is just a collection of units connected together; the properties of the network are determined by its topology and the properties of the “neurons.”

> Excerpt from [Artificial Intelligence: A Modern Approach](https://www.pearson.com/us/higher-education/program/Russell-Artificial-Intelligence-A-Modern-Approach-3rd-Edition/PGM156683.html)

![](https://github.com/justweiss/Neural-Network/blob/main/neuron.png)

This program implements a feed forward neural network with no back propagation. The weights and the bias's are set below and never change. It takes in a csv file with examples and outputs a value between 0 and 1. It creates a neural network with three layers the input layer with 10 input nodes, the hidden layer with 10 nodes and the output layer with 1 node.

	RUN: python3 NeuralNetwork.py

> NOTE: This program requires pandas and numpy, without these libraries the program will not run.
