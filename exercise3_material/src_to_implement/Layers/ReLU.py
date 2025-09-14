import numpy as np
from .Base import BaseLayer
import copy

class ReLU(BaseLayer):
    def __init__(self):
        self.input_tensor = None
        self.trainable = False #Non trainable layer
    
    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        self.output_tensor = np.maximum(0, input_tensor)#ReLU function
        return self.output_tensor

    def backward(self, error_tensor):
        # Gradient of ReLU function
        relu_grad = (self.output_tensor > 0).astype(error_tensor.dtype)
        # element-wise multiplication with incoming error tensor
        gradient_tensor = error_tensor * relu_grad
        return gradient_tensor