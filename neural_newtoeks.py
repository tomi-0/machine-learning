#---------------------- How Neural Networks are formed ----------------------#
# We have multiple layer of neurons
# First layer (training and testing data) is put in
# Final/output layer gives us our result e.g. classification
# can also give us probabilities of each classification

# In between we have hidden layers whch add complexity and sophistication to model
# In a normal feed-forward NN we connect every neuron of 1 layer to every neuron of the next layer, connected alayer by layer till output layer
# the neurons and connections are random to begin with

# E.g. inputs of 28 x 28px handwritten digits
# We would have a total of 784 pixels so and input layer with 784 neurons 

# All these neurons and connections need to be changed in a certain way so theyfit the training datat then we can use these to classify a new unknown handwritten digit

#---------------------- How Neurons are formed ----------------------#
# Each neuron has an input can be either input training data or output of a nother neuron
# each neuron has a certain activation function (a) which determines what happens to the input, processes the input, and will tell us how activated out neuron is

# The oudated perceptron model we take the input and subtract the bias to calculate the activation

# Esch neuron also has outputs(except output layer) 
# Every connection has a certain weight which determines how important the activation of one neuron is for the activation of the next neuron (NN does this)
# 

# Activation functions: 
# 1. sigmoid function - transforms the input value to a values between 0 and 1
# 2. ReLU - Rectified Linear Unit - if the input is negative returns 0 otherwise if +ve just return the input (max of 0 and input)


#---------------------- What is Gradient Descent? ----------------------#
# Algorithm that allows our neural network to be optimised
# We come up with a loss function from the probability of each neuron in the output layer which inidcates how incorrect our model is
# Not error, error is a percentage of misclassifcation
# our goal is to minimise the loss function

# Loss function's parameters would be all that makes up our neural netwokr e.g. weights, biases, outcomes and etc...
# We want to adjust weights and biases to reach the minimum point of the loss function graph (multidimensional graph)
# To roll down we want to go in the direction of steepest descent (gradient of loss function negated)
# We then take a small step in that direction adn repeat until we reach the local minimum

# To not get stuck at the local instead of global minimum we can try different starting points (randomising)

