import numpy as np

class Optimizer:
    def __init__(self):
        self.regularizer = None
    
    def add_regularizer(self, regularizer): #regaularizer method
        self.regularizer = regularizer #member variable regularizer

class Sgd(Optimizer):
    def __init__(self, learning_rate):
        super().__init__()
        self.learning_rate = learning_rate
    
    def calculate_update(self, weight_tensor, gradient_tensor):
        weight_tensor = np.asarray(weight_tensor)
        gradient_tensor = np.asarray(gradient_tensor)
        if self.regularizer is not None: # Apply weight shrinkage 
            shrinked_weights = weight_tensor - self.learning_rate * self.regularizer.calculate_gradient(weight_tensor)
        else:
            shrinked_weights = weight_tensor

        # gradient update to shrinked weights
        return shrinked_weights - self.learning_rate * gradient_tensor

        

class SgdWithMomentum(Optimizer):
    def __init__(self, learning_rate, momentum_rate):
        super().__init__()
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.velocity = None
    
    def calculate_update(self, weight_tensor, gradient_tensor):

        weight_tensor = np.asarray(weight_tensor)
        gradient_tensor = np.asarray(gradient_tensor)
        
        # Weight shrinkage if regularizer exists
        if self.regularizer is not None:
            shrinked_weights = weight_tensor - self.learning_rate * self.regularizer.calculate_gradient(weight_tensor)
        else:
            shrinked_weights = weight_tensor
            
        if self.velocity is None:
            self.velocity = np.zeros_like(weight_tensor)
        
        self.velocity = self.momentum_rate * self.velocity + self.learning_rate * gradient_tensor
    
        return shrinked_weights - self.velocity

class Adam(Optimizer):
    def __init__(self, learning_rate, mu, rho):
        super().__init__()
        self.learning_rate = learning_rate
        self.mu = mu              #  The exponential decay rate for the 1st moment estimates
        self.rho = rho            #  The exponential decay rate for the 2nd moment estimates
        self.m = None             # First moment(mean) estimate
        self.v = None             # Second moment(variance) estimate
        self.t = 0                # Time step counter
        self.epsilon = np.finfo(float).eps     # Smallest possible float positive number that machine can represent
    
    def calculate_update(self, weight_tensor, gradient_tensor):
        weight_tensor = np.asarray(weight_tensor)
        gradient_tensor = np.asarray(gradient_tensor)
        
        # Weight shrinkage if regularizer exists
        if self.regularizer is not None:
            shrinked_weights = weight_tensor - self.learning_rate * self.regularizer.calculate_gradient(weight_tensor)
        else:
            shrinked_weights = weight_tensor
       
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
        return shrinked_weights - update