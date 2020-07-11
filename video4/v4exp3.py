import numpy as np

#What is the size of the input coming in ie the size of a single sample
# [1,2,3,2.5] the size is 4 of the input
X=[[1,2,3,2.5],[2.0,5.0,-1.0,2.0],[-1.5,2.7,3.3,-0.8]];

#Initialization
np.random.seed(0)

class Layer_Dense(object):

    def __init__(self,n_inputs,n_neurons):
        #randn is a Gausian distribution bounded around 0, if its bigger than 1 then multiply with 0.10, The parameters are the shape
        self.weights = np.random.randn(n_inputs,n_neurons)
        #The first parameter is the shape of np.zeros
        self.biases = np.zeros((1,n_neurons))
    def forward(self,inputs):
        #The inputs is either the values of X or the output from the previous layer
        self.output = np.dot(inputs,self.weights) + self.biases

#parameter 1 is the input of the layer and parameter 2 is the output of the layer
layer1 = Layer_Dense(4,5)
#parameter 1 is the input of layer1 and has to be the same size as the output of layer1 ie 5
#parameter 2 can be of anysize
layer2 = Layer_Dense(5,2)

layer1.forward(X)
print("Layer 1 \n",layer1.output)
layer2.forward(layer1.output)
print("Layer 2 \n",layer2.output)
