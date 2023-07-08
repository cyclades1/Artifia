import sys
import numpy as np
import matplotlib

np.random.seed(0)

X = np.array([
    [1, 4, 3.4, 2.1],
    [5, 7, 6.1, 3.8],
    [3.1, 9.1, 2.6, 1.1]
])
Y =np.array([1,0,1])

class nueron_layer:
    def __init__(self, n_input, n_nueron):
        self.weight = 0.1* np.random.randn(n_input, n_nueron)
        self.bias = np.zeros((1, n_nueron))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weight) + self.bias
class activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class softmax:
    def forward(self, inputs):
        exp_value = np.exp(inputs- np.max(inputs, axis=1, keepdims=True))
        prob_value = exp_value/ np.sum(exp_value, axis=1, keepdims=True)
        self.output = prob_value

class loss:
    def calculate(self, output_pre, output_true):
        sample_losses = self.forward(output_pre, output_true)
        self.loss = np.mean(sample_losses)
        return self.loss

class entropy_loss(loss):
    def forward(self, output_pre, output_true):
        samples = len(output_pre)
        y_clipped = np.clip(output_pre,1e-7, 1-1e-7 )
        if len(output_true.shape)==1:
            correct_confidence = y_clipped[range(samples), output_true]
        elif len(output_true.shape)==2:
            correct_confidence = np.sum(output_pre*output_true, axis=1)
        negative_likelihoods = -np.log(correct_confidence)
        return negative_likelihoods
    
class accuracy:
    def calculate(self, output_pre, output_true):
        prediction = np.argmax(output_pre, axis=1)
        accuracy = np.mean(prediction==output_true)
        return accuracy


def main():
    layer1 = nueron_layer(len(X[0]), 5)
    activation1 = activation_ReLU()
    layer2 = nueron_layer(5, 2)
    activation2 = softmax()
    loss_function = entropy_loss()
    accuracy_function = accuracy()

    lowest_loss = 99999
    max_accuracy = 0

    best_layer1_weight = layer1.weight.copy()
    best_layer1_bias = layer1.bias.copy()
    best_layer2_weight = layer2.weight.copy()
    best_layer2_bias = layer2.bias.copy()

    for i in range(100000):

        layer1.weight += 0.05* np.random.randn(len(X[0]), 5)
        layer1.bias += 0.05* np.random.randn(1, 5)
        layer2.weight += 0.05* np.random.randn(5, 2)
        layer2.bias += 0.05* np.random.randn(1,2)

        layer1.forward(X)
        activation1.forward(layer1.output)
        layer2.forward(activation1.output)
        activation2.forward(layer2.output)
        loss = loss_function.calculate(activation2.output, Y)

        accuracy_value = accuracy_function.calculate(activation2.output, Y)


        if loss < lowest_loss:
            print("lowest loss found : {} at iteration: {} with accuracy : {}".format(loss, i, accuracy_value))
            best_layer1_weight = layer1.weight.copy()
            best_layer1_bias = layer1.bias.copy()
            best_layer2_weight = layer2.weight.copy()
            best_layer2_bias = layer2.bias.copy()
            lowest_loss = loss
            max_accuracy = accuracy_value
        else:
            layer1.weight = best_layer1_weight.copy()
            layer1.bias = best_layer1_bias.copy()
            layer2.weight = best_layer2_weight.copy()
            layer2.bias = best_layer2_bias.copy()


    print("minimum loss: {} with accuracy: {}%".format(lowest_loss, max_accuracy*100))

       
if __name__=="__main__":
    main()