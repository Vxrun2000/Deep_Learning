import numpy as np
from Layers.FullyConnected import FullyConnected

class RNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.trainable = True
        self.memorize = False

        self._hidden_state = None

        #Two FullyConnected layers
        self.fc_hidden = FullyConnected(input_size + hidden_size, hidden_size)
        self.fc_output = FullyConnected(hidden_size, output_size)

        self.optimizer = None

        # Caches for backward pass
        self._cache_inputs = []
        self._cache_hidden_states = []
        self._cache_concatenated = []

    @property #weights property
    def weights(self):
        return self.fc_hidden.weights
 
    @weights.setter
    def weights(self, new_weights):
        self.fc_hidden.weights = new_weights
    
    @property #gradient_weights property
    def gradient_weights(self):
        return self.fc_hidden._gradient_weights

    def initialize(self, weights_initializer, bias_initializer):
        self.fc_hidden.initialize(weights_initializer, bias_initializer)
        self.fc_output.initialize(weights_initializer, bias_initializer)
    
    def calculate_regularization_loss(self):
        total_loss = 0.0
        
        # FC layers regularization
        for layer in [self.fc_hidden, self.fc_output]:
            if hasattr(layer, 'optimizer') and layer.optimizer is not None:
                if hasattr(layer.optimizer, 'regularizer') and layer.optimizer.regularizer is not None:
                    total_loss += layer.optimizer.regularizer.norm(layer.weights)
        
        # RNN optimizer for regularization
        if hasattr(self, 'optimizer') and self.optimizer is not None:
            if hasattr(self.optimizer, 'regularizer') and self.optimizer.regularizer is not None:
                total_loss += self.optimizer.regularizer.norm(self.weights)
                
        return total_loss

    def forward(self, input_tensor): #Forward method treating batch dimension as time dimension.
        batch_size = input_tensor.shape[0]

        # Initialize hidden state
        if self._hidden_state is None or not self.memorize:
            self._hidden_state = np.zeros((1, self.hidden_size))

        # Clear caches if not memorizing
        if not self.memorize:
            self._cache_inputs.clear()
            self._cache_hidden_states.clear()
            self._cache_concatenated.clear()

        outputs = []
        for t in range(batch_size):
            # Get input for this time step
            x_t = input_tensor[t:t+1]
            
            # Cache for backward pass
            self._cache_inputs.append(x_t.copy())
            self._cache_hidden_states.append(self._hidden_state.copy())

            # Concatenate input and previous hidden state
            concatenated = np.hstack([x_t, self._hidden_state])
            self._cache_concatenated.append(concatenated.copy())

            # Calculate new hidden state
            hidden_pre_activation = self.fc_hidden.forward(concatenated)
            self._hidden_state = np.tanh(hidden_pre_activation)

            # Compute output
            output_t = self.fc_output.forward(self._hidden_state)
            outputs.append(output_t)

        return np.vstack(outputs)

    def backward(self, error_tensor): #backpropagation through time (BPTT)
        
        batch_size = error_tensor.shape[0]
        
        # Initialize gradients
        grad_w_hidden = np.zeros_like(self.fc_hidden.weights)
        grad_w_output = np.zeros_like(self.fc_output.weights)

        # gradient for hidden state propagation
        grad_h_next = np.zeros((1, self.hidden_size))
        grad_inputs = []

        # BPTT
        for t in reversed(range(len(self._cache_inputs))):
            x_t = self._cache_inputs[t]
            h_prev = self._cache_hidden_states[t]
            concatenated = self._cache_concatenated[t]
            
            # Error for this time step
            error_t = error_tensor[t:t+1]

            # Backward through output layer
            self._disable_optimizer_temporarily(self.fc_output)
            grad_h_output = self.fc_output.backward(error_t)
            grad_w_output += self.fc_output._gradient_weights
            self._restore_optimizer(self.fc_output)

            # Total gradient hiddden state
            grad_h_total = grad_h_output + grad_h_next

            # backpropagation through time (BPTT).
            hidden_pre_activation = self.fc_hidden.forward(concatenated)
            h_t = np.tanh(hidden_pre_activation)

            # Gradient through tanh activation
            grad_tanh = (1 - h_t ** 2) * grad_h_total

            # Backward through hidden layer
            self._disable_optimizer_temporarily(self.fc_hidden)
            grad_concatenated = self.fc_hidden.backward(grad_tanh)
            grad_w_hidden += self.fc_hidden._gradient_weights
            self._restore_optimizer(self.fc_hidden)

            # Split gradients
            grad_input_t = grad_concatenated[:, :self.input_size]
            grad_h_next = grad_concatenated[:, self.input_size:]
            
            grad_inputs.insert(0, grad_input_t)

        # Set accumulated gradients for the layers
        self.fc_hidden._gradient_weights = grad_w_hidden
        self.fc_output._gradient_weights = grad_w_output

        # Apply optimizer updates if available
        if self.optimizer is not None:
            self.fc_hidden.weights = self.optimizer.calculate_update(
                self.fc_hidden.weights, self.fc_hidden._gradient_weights)
            self.fc_output.weights = self.optimizer.calculate_update(
                self.fc_output.weights, self.fc_output._gradient_weights)

            # Clear gradients after update
            self.fc_hidden._gradient_weights = None
            self.fc_output._gradient_weights = None

        # Clear caches if not memorizing
        if not self.memorize:
            self._cache_inputs.clear()
            self._cache_hidden_states.clear()
            self._cache_concatenated.clear()

        return np.vstack(grad_inputs)

    def _disable_optimizer_temporarily(self, layer): #Temporarily disable optimizer to avoid premature weight updates
        
        if hasattr(layer, 'optimizer'):
            layer._temp_optimizer = layer.optimizer
            layer.optimizer = None

    def _restore_optimizer(self, layer): #Restore previously disabled optimizer
        if hasattr(layer, '_temp_optimizer'):
            layer.optimizer = layer._temp_optimizer
            delattr(layer, '_temp_optimizer')