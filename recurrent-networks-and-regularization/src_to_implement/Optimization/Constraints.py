import numpy as np

class L2_Regularizer: #L2 regularizer class
    def __init__(self, alpha): #constructor
        self.alpha = alpha
    
    def norm(self, weights):
        return self.alpha * np.sum(weights ** 2)
    
    def calculate_gradient(self, weights):
        return  self.alpha * weights


class L1_Regularizer: #L1 regularizer class
    def __init__(self, alpha): #constructor
        self.alpha = alpha
    
    def norm(self, weights):
        return self.alpha * np.sum(np.abs(weights))
    
    def calculate_gradient(self, weights):
        return self.alpha * np.sign(weights)