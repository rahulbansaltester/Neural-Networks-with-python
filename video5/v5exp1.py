import numpy as np
# custom package specific to the nnfs series
import nnfs

# init sets the random seed
nnfs.init()
np.random.seed(0)

inputs = [0,2,-1,3.3,-2.7,1.1,2.2,-100]
output = []

for i in inputs:
    output.append(max(0,i))


print(output)



import numpy as np
import nnfs

#nnfs.init()
np.random.seed(0)


inputs = [0,2,-1,3.3,-2.7,1.1,2.2,-100]
output = []

for i in inputs:
    if i > 0:
        output.append(max(0,i))
    elif i<=0:
        output.append(0)

def create_data(points,classes):
    X = np.zeros((points*classes,2))
    y = np.zeros(points*classes,dtype='uint8')
    for class_number in range(classes):
        ix=range(points*class_number,points*(class_number+1))
        r = np.linspace(0.0,1,points)
        t=np.linspace(class_number*4,(class_number+1)*4,points)+np.random.randn(points)*0.2
        X[ix]=np.c_[r*np.sin(t*2.5),r*np.cos(t*2.5)]
        y[ix]=class_number
    return X,y

class Layer_Dense:
    def __init__(self,n_inputs,n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1,n_neurons))
    def forward(self,inputs):
        self.output = np.dot(inputs,self.weights) + self.biases

class  Activation_ReLU:
    def forward(self,inputs):
        self.output = np.maximum(0,inputs)

layer1 = Layer_Dense(4,5)
layer2 = Layer_Dense(5,2)

layer1.forward(X)
layer2.forward(layer1.output)

import matplotlib.pyplot as plt

print("here")
X,y = create_data(100,3)

plt.scatter(X[:,0],X[:,1])
plt.show()

plt.scatter(X[:,0],X[:,1],c=y,cmap="brg")
plt.show()
