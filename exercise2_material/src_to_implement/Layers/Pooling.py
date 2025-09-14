# Layers/Pooling.py
import numpy as np
from .Base import BaseLayer

class Pooling(BaseLayer): #Pooling reduces the dimensionality of input
    def __init__(self, stride_shape, pooling_shape):
        super().__init__()
        self.trainable = False
        
        
        if isinstance(stride_shape, (int, float)): #Stride shape
            self.stride = (int(stride_shape), int(stride_shape))
        else:
            self.stride = tuple(int(s) for s in stride_shape)
          
        if isinstance(pooling_shape, (int, float)): #Pooling shape
            self.pooling_shape = (int(pooling_shape), int(pooling_shape))
        else:
            self.pooling_shape = tuple(int(s) for s in pooling_shape)
        
        
        self.input_tensor = None #For gradient computation
        self.max_indices = None #Store where max
    
    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        batch_size, channels, input_height, input_width = input_tensor.shape
        
        pool_h, pool_w = self.pooling_shape
        stride_h, stride_w = self.stride
        
        #output with valid padding
        output_height = (input_height - pool_h) // stride_h + 1
        output_width = (input_width - pool_w) // stride_w + 1
        
        output_tensor = np.zeros((batch_size, channels, output_height, output_width))
        
        # Indices of maximum values for backward pass
        self.max_indices = np.zeros((batch_size, channels, output_height, output_width, 2), dtype=int)
        
        for batch in range(batch_size):
            for channel in range(channels):
                for out_h in range(output_height):
                    for out_w in range(output_width):
                        # Input region
                        start_h = out_h * stride_h
                        start_w = out_w * stride_w
                        end_h = start_h + pool_h
                        end_w = start_w + pool_w
                        
                        # Pooling region
                        pool_region = input_tensor[batch, channel, start_h:end_h, start_w:end_w]
                        
                        # Maximum value and its position
                        max_val = np.max(pool_region)
                        max_pos = np.unravel_index(np.argmax(pool_region), pool_region.shape)
                        
                        # output
                        output_tensor[batch, channel, out_h, out_w] = max_val
                        
                        # Global indices of maximum position
                        self.max_indices[batch, channel, out_h, out_w, 0] = start_h + max_pos[0]
                        self.max_indices[batch, channel, out_h, out_w, 1] = start_w + max_pos[1]
        
        return output_tensor
    
    def backward(self, error_tensor):
        batch_size, channels, input_height, input_width = self.input_tensor.shape
        input_gradient = np.zeros_like(self.input_tensor)
        
        output_height, output_width = error_tensor.shape[2], error_tensor.shape[3]
        
        for batch in range(batch_size):
            for channel in range(channels):
                for out_h in range(output_height):
                    for out_w in range(output_width):
                        # Store maximum position
                        max_h = self.max_indices[batch, channel, out_h, out_w, 0]
                        max_w = self.max_indices[batch, channel, out_h, out_w, 1]
                        input_gradient[batch, channel, max_h, max_w] += error_tensor[batch, channel, out_h, out_w]
        
        return input_gradient
