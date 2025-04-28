import numpy as np
from nn.layers.base import Layer

# Flatten Layer
class Flatten(Layer):
    def __init__(self):
        self.input_shape = None
    
    def forward(self, x):
        self.input_shape = x.shape
        return x.reshape(x.shape[0], -1)
    
    def backward(self, grad_output):
        return grad_output.reshape(self.input_shape)
    
    def parameters(self):
        return []
    
    def get_gradients(self):
        return []

# ReLU Layer
class ReLU(Layer):
    def __init__(self):
        self.input = None
    
    def forward(self, x):
        self.input = x
        return np.maximum(0, x)
    
    def backward(self, grad_output):
        grad_input = grad_output.copy()
        grad_input[self.input < 0] = 0
        return grad_input
    
    def parameters(self):
        return []
    
    def get_gradients(self):
        return []