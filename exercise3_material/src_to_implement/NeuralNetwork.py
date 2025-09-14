import copy
from Layers.Initializers import Xavier, Constant

class NeuralNetwork:
    def __init__(self, optimizer, weights_initializer=None, bias_initializer=None):
        #5 memeber variables
        self.optimizer = optimizer           
        self.loss = []                       
        self.layers = []                    
        self.data_layer = None               
        self.loss_layer = None
        self.input_tensor = None
        self.label_tensor = None 
        
        if weights_initializer is None:
            self.weights_initializer = Xavier()
        else:
            self.weights_initializer = weights_initializer
            
        if bias_initializer is None:
            self.bias_initializer = Constant(0.0)
        else:
            self.bias_initializer = bias_initializer

    @property #phase property
    def phase(self):
        return self._phase
    
    @phase.setter
    def phase(self, phase):
        self._phase = phase
        for layer in self.layers:
            if hasattr(layer, 'testing_phase'):
                layer.testing_phase = phase

    def forward(self): #Forward method
        self.input_tensor, self.label_tensor = self.data_layer.next()
        tensor = self.input_tensor
        for layer in self.layers:
            tensor = layer.forward(tensor)
        output = self.loss_layer.forward(tensor, self.label_tensor)#Finally pass through loss layer
        
        # Regularization loss from all trainable layers
        regularization_loss = 0
        for layer in self.layers:
            if hasattr(layer, 'trainable') and layer.trainable:
                if hasattr(layer, 'optimizer') and layer.optimizer is not None:
                    if hasattr(layer.optimizer, 'regularizer') and layer.optimizer.regularizer is not None:
                        if hasattr(layer, 'weights') and layer.weights is not None:
                            regularization_loss += layer.optimizer.regularizer.norm(layer.weights)
        
        return output + regularization_loss

    def backward(self): #Backward method
        tensor = self.loss_layer.backward(self.label_tensor)
        for layer in reversed(self.layers):
            tensor = layer.backward(tensor)
    
    def append_layer(self, layer):
        if hasattr(layer, 'trainable') and layer.trainable: 
           if hasattr(layer, 'initialize'):
                layer.initialize(self.weights_initializer, self.bias_initializer) 
           layer.optimizer = copy.deepcopy(self.optimizer)#Deep copy if layer is trainable.
        self.layers.append(layer)

    def train(self, iterations): #train method
        self.phase = False  # Set to training phase
        for _ in range(iterations):
            output = self.forward()
            self.backward()
            self.loss.append(output)

    def test(self, input_tensor): #test method
        self.phase = True  # Set to testing phase
        tensor = input_tensor
        for layer in self.layers:
            tensor = layer.forward(tensor)
        return tensor