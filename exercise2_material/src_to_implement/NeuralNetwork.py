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


        

    def forward(self):
        #data layer provides an input tensor and a label tensor upon calling next()
        self.input_tensor, self.label_tensor = self.data_layer.next()
        tensor = self.input_tensor

        #Forward pass
        for layer in self.layers:
            tensor = layer.forward(tensor)
        #Finally pass through loss layer
        output = self.loss_layer.forward(tensor, self.label_tensor)
        return output

    def backward(self):
        # Backward pass begins from loss layer
        tensor = self.loss_layer.backward(self.label_tensor)
        # Going opposite to forward pass
        for layer in reversed(self.layers):
            tensor = layer.backward(tensor)
    
    def append_layer(self, layer):
        if hasattr(layer, 'trainable') and layer.trainable: 
           if hasattr(layer, 'initialize'):
                layer.initialize(self.weights_initializer, self.bias_initializer) 
           layer.optimizer = copy.deepcopy(self.optimizer)#Deep copy if layer is trainable.
        self.layers.append(layer)

    def train(self, iterations): #trains and stores the loss for each iteration
        for _ in range(iterations):
            output = self.forward()
            self.backward()
            self.loss.append(output)

    def test(self, input_tensor):#propagates the input tensor through the network and returns the prediction of the last layer.
      tensor = input_tensor
      for layer in self.layers:
        tensor = layer.forward(tensor)
      return tensor
