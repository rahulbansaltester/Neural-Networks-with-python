# Model of one Layer of Neurons using Numpy
# Formula output = (input 0 * weight 0)+...+(input n * weight n)+bias

import numpy as np 

inputs =[1,2,3,2.5]

weights =	[[0.2,0.8,-0.5,1.0],
			[0.5,-0.91,0.26,-0.5],
			[-0.26,-0.27,0.17,0.87]]

biasis =[2,3,0.5]

output = np.dot(weights,inputs) + biasis

print(output)