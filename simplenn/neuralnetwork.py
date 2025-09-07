import random
from engine import Data 

class Neuron:
    def __init__(self, neuron_inputs):
        self.weights = [Data(random.uniform(-1, 1), op = 'const') for i in range(neuron_inputs)]
        self.bias = Data(0, op = 'const - BIAS')   

    def __call__(self, inputs, is_relu = False):
        total = sum((wi*xi for wi, xi in zip(self.weights, inputs)), self.bias)
        return total.relu() if is_relu else total
    
class Layer:
    def __init__(self, neuron_inputs, number_of_neurons, is_relu = False):
        self.neurons = [Neuron(neuron_inputs) for _ in range(number_of_neurons)]
        self.is_relu = is_relu
    def __call__(self, inputs):
        outputs = [neuron(inputs, self.is_relu) for neuron in self.neurons]
        return outputs
    
class NeuralNetwork:
    def __init__(self, input, layers):
        layers_plus_input = [input] + layers
        self.layers = [Layer(layers_plus_input[i], layers_plus_input[i + 1], is_relu =  i != (len(layers) - 1)) for i in range(0, len(layers) - 1)]
    def __call__(self, inputs):
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs