# Model of one Neuron with 3 input
# Formula output = (input 0 * weight 0)+...+(input n * weight n)+bias

input =[1,2,3]
weight =[0.2,0.8,-0.5]
bias =2


output = input[0]*weight[0] + input[1]*weight[1] + input[2]*weight[2] + bias

print(output) 