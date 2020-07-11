import numpy as np
# custom package specific to the nnfs series
import nnfs
from nnfs.datasets import spiral_data
# init sets the random seed in place of np.random.seed(0) to avoid strange values in different systems
nnfs.init()
#What is the size of the input coming in ie the size of a single sample
# [1,2,3,2.5] the size is 4 of the input
# X is the featuresets in machine learning
X=[[1,2,3,2.5],[2.0,5.0,-1.0,2.0],[-1.5,2.7,3.3,-0.8]];
# y is the classification
X,y = spiral_data(100,3)
class Layer_Dense(object):
    def __init__(self,n_inputs,n_neurons):
        #randn is a Gausian distribution bounded around 0, if its bigger than 1 then multiply with 0.10, The parameters are the shape
        self.weights = np.random.randn(n_inputs,n_neurons)
        #The first parameter is the shape of np.zeros
        self.biases = np.zeros((1,n_neurons))
    def forward(self,inputs):
        #The inputs is either the values of X or the output from the previous layer
        self.output = np.dot(inputs,self.weights) + self.biases

class  Activation_ReLU:
# Forward pass for activation function
    def forward(self,inputs):
        self.output = np.maximum(0,inputs)


#parameter 1 is the input of the layer, and parameter 2 is the output of the layer
layer1 = Layer_Dense(2,5)
# activation function which is a ReLU object
activation1 = Activation_ReLU()
layer1.forward(X)
print("Layer1 \n",layer1.output)
activation1.forward(layer1.output)
#The optimizer tweaks the  Zero values, otherwise biases will be tweaked to get non zero numbers.
print("Activation\n",activation1.output)
