import numpy as np


class TanH:
    def __init__(self): #constructor
        self.trainable = False
        self.activations = None
    
    def forward(self, input_tensor): #Apply tanh activation function
        self.activations = np.tanh(input_tensor)
        return self.activations
    
    def backward(self, error_tensor): ##Derivative of tanh
        derivative = 1 - np.square(self.activations)
        return error_tensor * derivative