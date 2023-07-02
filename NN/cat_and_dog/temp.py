import sys
import numpy as np
import matplotlib

np.random.seed(0)

X = [
    [1, 4, 3.4, 2.1],
    [5, 7, 6.1, 3.8],
    [3.1, 9.1, 2.6, 1.1]
]

class nueron_layer:
    def __init__(self, n_input, n_nueron):
        self.weight = 0.1* np.random.randn(n_input, n_nueron)
        self.bias = np.zeros((1, n_nueron))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weight) + self.bias
class activation_rl:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


layer1 = nueron_layer(len(X[0]), 5)
activation1 = activation_rl()
layer2 = nueron_layer(5, 2)
activation2 = activation_rl()
layer1.forward(X)
activation1.forward(layer1.output)
layer2.forward(activation1.output)
activation2.forward(layer2.output)
print(activation2.output)