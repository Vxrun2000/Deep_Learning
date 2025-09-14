import numpy as np
from .Base import BaseLayer

class Flatten(BaseLayer):
    def __init__(self):
        super().__init__()
        self.trainable = False  # Non trainable layer
        self.input_shape = None  # Store input shape for backward pass

    def forward(self, input_tensor):
        self.input_shape = input_tensor.shape #Input tensor shape : (batch_size, no_of_features)#(batch_size, channels, height, width)
        batch_size = input_tensor.shape[0]
        return input_tensor.reshape(batch_size, -1) #-1 here multiplies all remaining dimensions.(batch_size, flatten_layer)

    def backward(self, error_tensor):
        return error_tensor.reshape(self.input_shape) #Reverses the flattening operation.
