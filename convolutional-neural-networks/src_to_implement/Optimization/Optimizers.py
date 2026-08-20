
import numpy as np

class Sgd:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
    
    def calculate_update(self, weight_tensor, gradient_tensor):
        return weight_tensor - self.learning_rate * gradient_tensor
        

class SgdWithMomentum:
    def __init__(self, learning_rate, momentum_rate):
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.velocity = None
    
    def calculate_update(self, weight_tensor, gradient_tensor):
        if self.velocity is None:
            self.velocity = np.zeros_like(weight_tensor) # Initialize velocity
        
        self.velocity = self.momentum_rate * self.velocity + self.learning_rate * gradient_tensor # Update velocity

        return weight_tensor - self.velocity #Update weights


class Adam:
    def __init__(self, learning_rate, mu, rho):
        self.learning_rate = learning_rate
        self.mu = mu              #  The exponential decay rate for the 1st moment estimates
        self.rho = rho            #  The exponential decay rate for the 2nd moment estimates
        self.m = None             # First moment(mean) estimate
        self.v = None             # Second moment(variance) estimate
        self.t = 0                # Time step counter
        self.epsilon = np.finfo(float).eps     # Smallest possible float positive number that machine can represent
    
    def calculate_update(self, weight_tensor, gradient_tensor):
       
        if self.m is None: 
            self.m = np.zeros_like(weight_tensor) #Initialize m
        if self.v is None:
            self.v = np.zeros_like(weight_tensor) #Initialize v
        
        
        self.t += 1 # Increment time step
        
        
        self.m = self.mu * self.m + (1 - self.mu) * gradient_tensor #Update m
        
        
        self.v = self.rho * self.v + (1 - self.rho) * (gradient_tensor ** 2) #Update v
        
        
        m_hat = self.m / (1 - self.mu ** self.t) #Bias corrected m
        
        
        v_hat = self.v / (1 - self.rho ** self.t) #Bias corrected v
        
        # Update weights
        update = self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        return weight_tensor - update