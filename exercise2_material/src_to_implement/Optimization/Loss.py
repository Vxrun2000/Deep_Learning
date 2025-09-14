import numpy as np

class CrossEntropyLoss:
    def __init__(self):
        self.prediction_tensor = None
        self.label_tensor = None

    def forward(self, prediction_tensor, label_tensor): #Forward method returns loss
        self.prediction_tensor = prediction_tensor
        self.label_tensor = label_tensor

        epsilon = np.finfo(float).eps #smallest possible positive number that the float datatype can represent and to avoid log(0)
        
        loss = -np.sum(label_tensor * np.log(prediction_tensor + epsilon))
        return loss

    def backward(self, label_tensor): #Returns gradient of the cross-entropy loss w.r.t predicted probabilities
        epsilon = np.finfo(float).eps
        return -label_tensor / (self.prediction_tensor + epsilon)
