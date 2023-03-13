import numpy as np
import nnfs

np.random.seed(0)

X =    [[1,2,3,2.5],
        [2.0,5.0,-1.0,2.0],
        [-1.5,2.7,3.3,-0.8]]


class Dense_Layer:
    def __init__(self,n_inputs,n_neurons) :
        self.weights = 0.1 * np.random.randn(n_inputs,n_neurons)
        self.biases = np.zeros((1,n_neurons))
    def forward(self,inputs):
        self.output = np.dot(inputs,self.weights) + self.biases

class Activation_RELU:
    def forward(self,inputs):
        self.output = np.maximum(0,input)


layer1 = Dense_Layer(4,5)
layer2 = Dense_Layer(5,1)

layer1.forward(X)
# print(layer1.output)
layer2.forward(layer1.output)
print(layer2.output)
        


# print(0.1*np.random.randn(4,3))
# weights = [[0.2,0.8,-0.5 ,1.0],
#           [0.5,-0.91,0.26,-0.5],
#           [-0.26,-0.27,0.17,0.87]]
# bias = [2,3,0.5]
# weights2 = [[0.1,-0.14,0.5],
#             [-0.5,0.12,-0.33],
#             [-0.44,0.73,-0.13]]
# bias2 = [-1,2,-0.5]
# layer1_output = np.dot(inputs,np.array(weights).T) + bias
# layer2_output = np.dot(layer1_output,np.array(weights2).T) + bias2
# print(layer2_output)
