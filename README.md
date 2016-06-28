# NeuralNet
This program trains a neural network to recognize letters from the Letter Recognition dataset at the UCI
machine-learning repository http://archive.ics.uci.edu/ml/datasets/Letter+Recognition.  The 20,000 item
dataset is divided into equal sized training and testing sets which are standardized.  The network trains 
until it achieves perfect accuracy catagorizing the training set or it reaches a set maximum number of 
training epochs.  During each epoch the network updates its weights using back-propogation with stochastic 
gradient descent.  After each training epoch the network checks its prediction accuracy against both the 
training and the testing set.
