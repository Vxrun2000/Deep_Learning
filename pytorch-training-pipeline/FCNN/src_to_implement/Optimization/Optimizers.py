import numpy as np

class Sgd: #Class
    def __init__(self, learning_rate: float):#Constructor
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor: np.ndarray, gradient_tensor: np.ndarray):#basic gradient descent update scheme.
        return weight_tensor - self.learning_rate * gradient_tensor