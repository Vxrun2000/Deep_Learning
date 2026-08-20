import numpy as np


class Sigmoid:
    def __init__(self):
        self.trainable = False
        self.activations = None
    
    def forward(self, input_tensor):
        # Sigmoid activation
        # Use np.clip to prevent overflow for very large negative values
        clipped_input = np.clip(input_tensor, -500, 500)
        self.activations = 1.0 / (1.0 + np.exp(-clipped_input))
        return self.activations
    
    def backward(self, error_tensor):
        # Sigmoid derivative
        # Since we stored activations, we can use: activations * (1 - activations)
        derivative = self.activations * (1 - self.activations)
        return error_tensor * derivative