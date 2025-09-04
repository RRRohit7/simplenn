class Data:
    def __init__(self, value):
        self.value = value
    def __add__(self, other):
        if isinstance(other, (int, float)):
            return Data(self.value + other)
        return Data(self.value + other.value)
    def __radd__(self, other):
        return self + other
    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Data(self.value * other)
        return Data(self.value * other.value)
    def __rmul__(self, other):
        return self * other    
    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return Data(self.value / other)
        return Data(self.value / other.value)
    def __rtruediv__(self, other):
        return Data(other / self.value)
    def __mod__(self, other):
        if isinstance(other, (int, float)):
            return Data(self.value % other)
        return Data(self.value % other.value)
    def __rmod__(self, other):
        return Data(other % self.value)
    def __str__(self):
        return f"Data: {self.value}"