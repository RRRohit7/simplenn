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
        output = Data(self.value ** other, (self,), '**')
        def backward():
            self.grad = other * (self.value ** (other - 1)) * output.grad
        output.backward = backward
        return output
    def __str__(self):
        return f"Data: {self.value}, operation: {self.op}"