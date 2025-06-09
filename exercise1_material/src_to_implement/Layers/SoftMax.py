import numpy as np

class SoftMax:
    def __init__(self):
        self.trainable = False #Non trainable layer
        self.prediction_tensor = None

    def forward(self, input_tensor): #returns the estimated class probabilities for each row representing an element of the batch.
        # Shift input to get numerical stability
        shift_input = input_tensor - np.max(input_tensor, axis=1, keepdims=True)
        exp_tensor = np.exp(shift_input)
        self.prediction_tensor = exp_tensor / np.sum(exp_tensor, axis=1, keepdims=True)
        return self.prediction_tensor

    def backward(self, error_tensor):
        # Computation of gradient (Jacobian-vector product)
        batch_size, n_classes = self.prediction_tensor.shape
        dot_product = np.sum(error_tensor * self.prediction_tensor, axis=1, keepdims=True)
        grad_input = self.prediction_tensor * (error_tensor - dot_product)
        return grad_input #Serves as error tensor for the previous layer.
