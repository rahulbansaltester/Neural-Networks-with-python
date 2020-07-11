import numpy as np

#Output each array within the inputs array is a row
inputs = [[1,2,3,2.5],[2.0,5.0,-1.0,2.0],[-1.5,2.7,3.3,-1.8]]

weights = [[0.2,0.8,-0.5,1.0],[0.5,-0.91,0.26,-0.5],[-0.26,-0.27,0.17,0.87]]

biases = [2,3,0.5]

# weights matrix is transposed using the .T to the weights array so as to do matrix multiplication
output=np.dot(inputs,np.array(weights).T)+biases

print(output)
