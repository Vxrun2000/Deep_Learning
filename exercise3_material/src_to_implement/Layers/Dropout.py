import numpy as np


class Dropout:
    def __init__(self, probability): #constructor 
        self.probability = probability
        self.trainable = False
        self.testing_phase = False
        self.mask = None
    
    def forward(self, input_tensor):
        if self.testing_phase: #input tensor unchanged
            return input_tensor
        else:
            self.mask = np.random.rand(*input_tensor.shape) < self.probability # Create random mask
            output = input_tensor * self.mask * (1.0 / self.probability) # Apply mask and scale
            
            return output
    
    def backward(self, error_tensor):
        if self.testing_phase: #error unchanged
            return error_tensor
        else:
            return error_tensor * self.mask * (1.0 / self.probability)#Apply mask and scale