import math
from enum import Enum

class Data:
    def __init__(self, value, children=(), op=''):
        self.value = value
        self.children = children
        self.op = op
        self.backward = lambda: None
        self.grad = 0.0

    def __add__(self, other):
        other = other if isinstance(other, Data) else Data(other, op='const')   
        output = Data(self.value + other.value, (self, other), '+')
        def backward():
            self.grad = output.grad
            other.grad = output.grad
        output.backward = backward
        return output   
    def __radd__(self, other):
        return self + other
    def __neg__(self):
        return self * -1
    def __sub__(self, other):
        return self + (-other)
    def __rsub__(self, other):
        return other + (-self)
    def relu(self):
        output = Data(self.value if self.value > 0 else 0, (self,), 'ReLU')
        def backward():
            self.grad = (output.value > 0) * output.grad
        output.backward = backward
        return output
    def sigmoid(self):
        x = self.value
        sigmoid = Data(1 / ( 1 + math.exp(x * -1)), (self,), 'Sigmoid')
        def backward():
             self.grad = sigmoid.value * (1 - sigmoid.value) * sigmoid.grad
        sigmoid.backward = backward
        return sigmoid
    def __mul__(self, other):
        other = other if isinstance(other, Data) else Data(other, op='const')
        output = Data(self.value * other.value, (self, other), '*')
        def backward():
            self.grad = other.value * output.grad
            other.grad = self.value * output.grad
        output.backward = backward
        return output
    def __rmul__(self, other):
        return self * other    
    def __pow__(self, other):
        assert isinstance(other, (int, float))
        other = Data(other, op='const')
        output = Data(self.value ** other.value, (self,), '**')
        def backward():
            self.grad = other.value * (self.value ** (other.value - 1)) * output.grad
            # other.grad = math.log(self.value) * (self.value ** other.value) * output.grad
        output.backward = backward
        return output
    def __truediv__(self, other):
        return self * other**-1
    def __rtruediv__(self, other):
        return other * self**-1 
    def backpropagation(self):
        order = []
        visited = set()
        def topo(node):
            if node not in visited:
                visited.add(node)
                for child in node.children:
                    topo(child)
                order.append(node)
        topo(self)
        self.grad = 1.0 #output gradient is 1
        for node in reversed(order):
            node.backward()
    
    def __str__(self):
        return f"Data: {self.value}, operation: {self.op}"

class LossType(Enum):
    MSE = 1 # only implementing Mean Square Error as of now

class Loss:
    def __init__(self, type: LossType):
        self.type = type
    
    def __call__(self, expected, original):
        assert self.type == LossType.MSE, "Only Supporting Mean Square Error loss for now"
        loss = sum([(e - o)**2 for e, o in zip(expected, original)]) / len(original)
        return loss
        


