import math
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
        output = Data(self.value ** other.value, (self, other), '**')
        def backward():
            self.grad = other.value * (self.value ** (other.value - 1)) * output.grad
            other.grad = math.log(self.value) * (self.value ** other.value) * output.grad
        output.backward = backward
        return output
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