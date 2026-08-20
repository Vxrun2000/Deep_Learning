import numpy as np
from .Base import BaseLayer
import copy

class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__()  # Super constructor
        self.input_size = input_size
        self.output_size = output_size
        self.trainable = True  # Trainable layer
        
        self.weights = np.random.rand(input_size + 1, output_size)  # weights
        
        self._gradient_weights = None
        self._gradient_bias = None
        
        # For optimizer
        self.optimizer = None

    def initialize(self, weights_initializer, bias_initializer): #initialize method
        fan_in = self.input_size
        fan_out = self.output_size
        
        # Initialize weights (excluding bias)
        weights_shape = (self.input_size, self.output_size)
        self.weights[:-1, :] = weights_initializer.initialize(weights_shape, fan_in, fan_out)
        
        # Initialize bias separately (last row of weights matrix)
        bias_shape = (1, self.output_size)
        self.weights[-1:, :] = bias_initializer.initialize(bias_shape, fan_in, fan_out)

    def forward(self, input_tensor):  # forward method
        bias_term = np.ones((input_tensor.shape[0], 1))
        self._input_tensor = np.hstack((input_tensor, bias_term))  # input tensor is a matrix with input size columns and batch size rows.
        return np.dot(self._input_tensor, self.weights)

    def backward(self, error_tensor):
        grad_w = np.dot(self._input_tensor.T, error_tensor)

        # Store gradient weights for optimizer
        self._gradient_weights = grad_w

        # Compute input gradient (exclude bias row from weights)
        error_backprop = np.dot(error_tensor, self.weights[:-1, :].T)

        # Apply optimizer if available
        if hasattr(self, 'optimizer') and self.optimizer is not None:
            self.weights = self.optimizer.calculate_update(self.weights, self._gradient_weights)
            
        return error_backprop