import numpy as np
from Layers.Helpers import compute_bn_gradients


class BatchNormalization:
    def __init__(self, channels): #constructor
        self.channels = channels
        self.trainable = True
        self.testing_phase = False
        self.optimizer = None
        self.bias_optimizer = None
        self.gradient_weights = None
        self.gradient_bias = None
        self.moving_mean = None
        self.moving_var = None
        self.momentum = 0.8
        self.input_shape = None
        self.batch_mean = None
        self.batch_var = None
        self.normalized_input = None
        self.input_tensor_cache = None
        self.eps = 1e-11
        self.initialize()
    
    def initialize(self, weights_initializer=None, bias_initializer=None):
        # Initialize weights as 1 and bias as 0
        self.weights = np.ones(self.channels)
        self.bias = np.zeros(self.channels)
    
    def reformat(self, tensor):# Reformat image like tensor(4D) to its vector variant(2D)
        if tensor.ndim == 4:  
            self.input_shape = tensor.shape
            batch_size, channels, height, width = tensor.shape
            return tensor.transpose(0, 2, 3, 1).reshape(-1, channels)
        elif tensor.ndim == 2 and self.input_shape is not None:
            batch_size, channels, height, width = self.input_shape
            reshaped = tensor.reshape(batch_size, height, width, channels)
            return reshaped.transpose(0, 3, 1, 2)
        else:
            return tensor
    
    def forward(self, input_tensor):
        self.input_tensor_cache = input_tensor.copy() # Cache input for backward pass
        
        # Reformat if 4D
        is_conv = input_tensor.ndim == 4
        if is_conv:
            input_tensor = self.reformat(input_tensor)
        
        if self.testing_phase:
            
            if self.moving_mean is None or self.moving_var is None:
                mean = np.mean(input_tensor, axis=0)
                var = np.var(input_tensor, axis=0)
            else:
                mean = self.moving_mean
                var = self.moving_var
        else:
            # Compute mean and variance during training
            mean = np.mean(input_tensor, axis=0)
            var = np.var(input_tensor, axis=0)
            
            # Update moving averages
            if self.moving_mean is None:
                self.moving_mean = mean.copy()
                self.moving_var = var.copy()
            else:
                self.moving_mean = self.momentum * self.moving_mean + (1 - self.momentum) * mean
                self.moving_var = self.momentum * self.moving_var + (1 - self.momentum) * var
            
            # Cache for backward pass
            self.batch_mean = mean
            self.batch_var = var
        
        # Normalize input
        self.normalized_input = (input_tensor - mean) / np.sqrt(var + self.eps)
        
        # Apply scale and shift
        output = self.weights * self.normalized_input + self.bias
        
        # Reformat back if needed
        if is_conv:
            output = self.reformat(output)
        
        return output
    
    def backward(self, error_tensor):
        is_conv = error_tensor.ndim == 4
        if is_conv:
            error_tensor = self.reformat(error_tensor)
            input_tensor = self.reformat(self.input_tensor_cache)
        else:
            input_tensor = self.input_tensor_cache
        
        
        self.gradient_weights = np.sum(error_tensor * self.normalized_input, axis=0)
        self.gradient_bias = np.sum(error_tensor, axis=0)
        
        # Update parameters with optimizers
        if self.optimizer is not None:
            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)
        
        if self.bias_optimizer is not None:
            self.bias = self.bias_optimizer.calculate_update(self.bias, self.gradient_bias)
        elif self.optimizer is not None:
            self.bias = self.optimizer.calculate_update(self.bias, self.gradient_bias)
        
        # Compute input gradients
        if self.batch_mean is None or self.batch_var is None:
            # If batch statistics(mean and variance) are not cached, compute them
            batch_mean = np.mean(input_tensor, axis=0)
            batch_var = np.var(input_tensor, axis=0)
        else:
            batch_mean = self.batch_mean
            batch_var = self.batch_var
            
        grad_input = compute_bn_gradients(error_tensor, input_tensor, self.weights, 
                                        batch_mean, batch_var, self.eps)
        
        # Reformat back if needed
        if is_conv:
            grad_input = self.reformat(grad_input)
        
        return grad_input