import random
from engine import Data, Loss, LossType

class Neuron:
    def __init__(self, neuron_inputs):
        self.weights = [Data(random.uniform(-1, 1), op = 'const') for i in range(neuron_inputs)]
        self.bias = Data(0, op = 'const - BIAS')   

    def __call__(self, inputs, is_relu = False):
        total = sum((wi*xi for wi, xi in zip(self.weights, inputs)), self.bias)
        return total.sigmoid() if is_relu else total
    
    def parameters(self):
        return self.weights + [self.bias]
    
    def set_grad_zero(self):
        for param in self.parameters():
            param.grad = 0.0
    
class Layer:
    def __init__(self, neuron_inputs, number_of_neurons, is_relu = False):
        self.neurons = [Neuron(neuron_inputs) for _ in range(number_of_neurons)]
        self.is_relu = is_relu
    def __call__(self, inputs):
        outputs = [neuron(inputs, self.is_relu) for neuron in self.neurons]
        return outputs if len(outputs) > 1 else outputs[0]
    
    def parameters(self):
        params = [p for neuron in self.neurons for p in neuron.parameters()]
        return params
    
    def set_grad_zero(self):
        for param in self.parameters():
            param.grad = 0.0
    
class NeuralNetwork:
    def __init__(self, input, layers):
        layers_plus_input = [input] + layers
        self.layers = [Layer(layers_plus_input[i], layers_plus_input[i + 1], is_relu =  (i == (len(layers) - 1))) for i in range(0, len(layers))]
    def __call__(self, inputs):
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs
    
    def parameters(self):
        params = [p for layer in self.layers for p in layer.parameters()]
        return params
    
    def set_grad_zero(self):
        for param in self.parameters():
            param.grad = 0.0
    
    def train(self, learning_rate, epochs, inputs, expected_outputs):
        for i in range(epochs):
            #forward pass
            predicted_outputs = [self(i) for i in inputs]
            lossfn = Loss(LossType.MSE)
            loss = lossfn(predicted_outputs, expected_outputs)
            #setting all param grads to zero before backpropagation
            self.set_grad_zero()
            #backward pass
            loss.backpropagation()
            for p in self.parameters():
                p.value += -learning_rate * p.grad
            print(f"Epoch: {i + 1} - Loss: {loss}")
