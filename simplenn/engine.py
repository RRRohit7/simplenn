class Data:
    def __init__(self, value, children=(), op=''):
        self.value = value
    def __add__(self, other):
        if isinstance(other, (int, float)):
            return Data(self.value + other, (self, Data(other), '+'))
        return Data(self.value + other.value, (self, other), '+')
    def __radd__(self, other):
        return self + other
    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Data(self.value * other, (self, Data(other), '*'))
        return Data(self.value * other.value, (self, other), '*')
    def __rmul__(self, other):
        return self * other    
    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return Data(self.value / other, (self, Data(other), '/'))
        return Data(self.value / other.value, (self, other), '/')
    def __rtruediv__(self, other):
        return Data(other / self.value, (other, self), '/')
    def __mod__(self, other):
        if isinstance(other, (int, float)):
            return Data(self.value % other, (self, Data(other), '%'))
        return Data(self.value % other.value, (self, other), '%')
    def __rmod__(self, other):
        return Data(other % self.value, (other, self), '%')
    def __str__(self):
        return f"Data: {self.value}"