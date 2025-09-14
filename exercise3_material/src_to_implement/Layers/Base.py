class BaseLayer: 
    def __init__(self): #Constructor with  no arguments
        self.trainable = False #Set trainable to false
        self.optimizer = None
        self.weights = None #Default weight parameter
        self.testing_phase = False #boolean member testing phase
