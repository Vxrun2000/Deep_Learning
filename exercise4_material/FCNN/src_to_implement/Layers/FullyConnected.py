import numpy as np
from .Base import BaseLayer
import copy

class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__()  # Super constructor
        self.input_size = input_size
        self.output_size = output_size
        self.trainable = True #Trainable layer
        

        self.weights = np.random.rand(input_size+1, output_size)  # weights

        self.bias = np.random.rand(output_size) #biases


        self._gradient_weights = None
        self._gradient_bias = None


    
    def forward(self, input_tensor): #forward method
        bias_term = np.ones((input_tensor.shape[0], 1)) 
        self._input_tensor = np.hstack((input_tensor, bias_term)) #input tensor is a matrix with input size columns and batch size rows.
        return np.dot(self._input_tensor, self.weights) 

    
    def backward(self, error_tensor):
    # Calculate gradients
      self.gradient_weights = np.dot(self._input_tensor.T, error_tensor)

    # Calculate gradient with respect to input (excluding bias)
      error_backprop = np.dot(error_tensor, self.weights.T[:, :-1])

    # Updating weights
      if self.optimizer:
        self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)

      return error_backprop
  
    @property #Properties offer a pythonic way of realizing getters and setters.
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, opt):
        self._optimizer = opt
