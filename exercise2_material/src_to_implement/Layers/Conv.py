# Layers/Conv.py
import numpy as np
from .Base import BaseLayer
from scipy import signal

class Conv(BaseLayer):
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        super().__init__()
        self.trainable = True
        
        # Store parameters
        self.stride_shape = stride_shape #Controls how kernel moves 
        self.convolution_shape = convolution_shape #Size of the kernel
        self.num_kernels = num_kernels #No of kernels(filters)
        
        
        if len(convolution_shape) == 2:
            self.conv_dim = 1  #[channels, kernel_size]
            self.input_channels, self.kernel_size = convolution_shape
            self.weights = np.random.uniform(0, 1, (num_kernels, self.input_channels, self.kernel_size)) #(0,1,(num_kernels*self.input_channels*self.kernel_size))
        elif len(convolution_shape) == 3:
            self.conv_dim = 2  # 2D convolution: [channels, height, width]
            self.input_channels, self.kernel_height, self.kernel_width = convolution_shape
            self.weights = np.random.uniform(0, 1, (num_kernels, self.input_channels, self.kernel_height, self.kernel_width))#(0, 1, (num_kernels*self.input_channels*self.kernel_height*self.kernel_width))
        
        
        
        
        if isinstance(stride_shape, (int, float)):
            if self.conv_dim == 1:#stride shape for 1D
                self.stride = (int(stride_shape),)
            else:
                self.stride = (int(stride_shape), int(stride_shape))#Stride shape for 2D
        else:
            self.stride = tuple(int(s) for s in stride_shape)#stride as tuple
        
        # Bias initilaization
        self.bias = np.random.uniform(0, 1, num_kernels) #One bias/kernel
        
        # To store gradients
        self._gradient_weights = None #grad wrt filter weights
        self._gradient_bias = None #grad wrt bias
        self.input_tensor = None #input for back prop
        self._optimizer = None #Optimizer for weights
        self._bias_optimizer = None #Optimizer for bias
    
    @property #Method to attribute
    def optimizer(self):
        return self._optimizer
    
    @optimizer.setter
    def optimizer(self, opt):
        self._optimizer = opt
        # Bias optimizer to prevent shape conflicts
        if opt is not None:
            import copy
            self._bias_optimizer = copy.deepcopy(opt)
    
    @property
    def gradient_weights(self):
        return self._gradient_weights
    
    @property
    def gradient_bias(self):
        return self._gradient_bias
    
    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        batch_size = input_tensor.shape[0] #No of samples
        input_channels = input_tensor.shape[1] #No of input feature maps
        
        if self.conv_dim == 1:
            input_length = input_tensor.shape[2] #No of sequential data points
            output_length = int(np.ceil(input_length / self.stride[0]))
            output_tensor = np.zeros((batch_size, self.num_kernels, output_length)) #Size of output tensor
            
            for batch in range(batch_size): #For each sample in a batch apply filter
                for kernel_idx in range(self.num_kernels):#Apply each filters
                    result = np.zeros(input_length)
                    
                    for channel in range(input_channels): #Convolution on each channel
                        corr_result = signal.correlate(self.input_tensor[batch, channel], 
                                                     self.weights[kernel_idx, channel], 
                                                     mode='same')
                        result += corr_result #Sum over correlation over input channels
                    
                    result += self.bias[kernel_idx] #Add bias to kernel
                    output_tensor[batch, kernel_idx] = result[::self.stride[0]][:output_length] #stride and truncate
         
        else:  # 2D convolution with height and width channels
            input_height, input_width = input_tensor.shape[2], input_tensor.shape[3]
            output_height = int(np.ceil(input_height / self.stride[0]))
            output_width = int(np.ceil(input_width / self.stride[1]))
            output_tensor = np.zeros((batch_size, self.num_kernels, output_height, output_width))
            
            for batch in range(batch_size):
                for kernel_idx in range(self.num_kernels):
                    result = np.zeros((input_height, input_width))
                    
                    for channel in range(input_channels):
                        corr_result = signal.correlate2d(self.input_tensor[batch, channel], 
                                                       self.weights[kernel_idx, channel], 
                                                       mode='same')
                        result += corr_result
                    
                    result += self.bias[kernel_idx]
                    output_tensor[batch, kernel_idx] = result[::self.stride[0], ::self.stride[1]][:output_height, :output_width]
        
        return output_tensor
    
    def backward(self, error_tensor):
        batch_size = self.input_tensor.shape[0]
        input_channels = self.input_tensor.shape[1]
        
        self._gradient_weights = np.zeros_like(self.weights)
        self._gradient_bias = np.zeros_like(self.bias)
        input_gradient = np.zeros_like(self.input_tensor)
        
        # Reshape error tensor if needed
        if error_tensor.ndim == 2:
            if self.conv_dim == 1:
                expected_shape = (batch_size, self.num_kernels, -1)
                error_tensor = error_tensor.reshape(expected_shape)
            else:
                
                input_height, input_width = self.input_tensor.shape[2], self.input_tensor.shape[3]
                output_height = int(np.ceil(input_height / self.stride[0]))
                output_width = int(np.ceil(input_width / self.stride[1]))
                error_tensor = error_tensor.reshape(batch_size, self.num_kernels, output_height, output_width)
        
        if self.conv_dim == 1:
            input_length = self.input_tensor.shape[2]
            
            for kernel_idx in range(self.num_kernels):
                self._gradient_bias[kernel_idx] = np.sum(error_tensor[:, kernel_idx, :]) #Bias gradient(Sum of all errors for a kernel)
            
            upsampled_error = np.zeros((batch_size, self.num_kernels, input_length))
            for batch in range(batch_size):
                for kernel_idx in range(self.num_kernels):
                    for i in range(error_tensor.shape[2]):
                        pos = i * self.stride[0] #reverse stride downsampling from forward pass
                        if pos < input_length: 
                            upsampled_error[batch, kernel_idx, pos] = error_tensor[batch, kernel_idx, i]#error values to original input positions
            
            for batch in range(batch_size): 
                for kernel_idx in range(self.num_kernels):
                    for channel in range(input_channels):
                        weight_grad = signal.correlate(self.input_tensor[batch, channel], 
                                                     upsampled_error[batch, kernel_idx], 
                                                     mode='same') #Correlation of input tensor and error tenosr to check how weight contributed to error
                        mid = len(weight_grad) // 2
                        start = mid - self.kernel_size // 2
                        end = start + self.kernel_size
                        self._gradient_weights[kernel_idx, channel] += weight_grad[start:end] #Center extraction and correlation gives us output greater than size of kernel
                        
                        input_grad = signal.convolve(upsampled_error[batch, kernel_idx], 
                                                   self.weights[kernel_idx, channel], 
                                                   mode='same')
                        input_gradient[batch, channel] += input_grad #input grad over all batches
        
        else:  # 2D
            input_height, input_width = self.input_tensor.shape[2], self.input_tensor.shape[3]
            
            for kernel_idx in range(self.num_kernels):
                self._gradient_bias[kernel_idx] = np.sum(error_tensor[:, kernel_idx, :, :])
            
            upsampled_error = np.zeros((batch_size, self.num_kernels, input_height, input_width))
            for batch in range(batch_size):
                for kernel_idx in range(self.num_kernels):
                    for i in range(error_tensor.shape[2]):
                        for j in range(error_tensor.shape[3]):
                            h_pos = i * self.stride[0]
                            w_pos = j * self.stride[1]
                            if h_pos < input_height and w_pos < input_width:
                                upsampled_error[batch, kernel_idx, h_pos, w_pos] = error_tensor[batch, kernel_idx, i, j]
            
            for batch in range(batch_size):
                for kernel_idx in range(self.num_kernels):
                    for channel in range(input_channels):
                        weight_grad = signal.correlate2d(self.input_tensor[batch, channel], 
                                                       upsampled_error[batch, kernel_idx], 
                                                       mode='same')
                        h_mid, w_mid = weight_grad.shape[0] // 2, weight_grad.shape[1] // 2
                        h_start = h_mid - self.kernel_height // 2
                        w_start = w_mid - self.kernel_width // 2
                        h_end = h_start + self.kernel_height
                        w_end = w_start + self.kernel_width
                        self._gradient_weights[kernel_idx, channel] += weight_grad[h_start:h_end, w_start:w_end]
                        
                        input_grad = signal.convolve2d(upsampled_error[batch, kernel_idx], 
                                                     self.weights[kernel_idx, channel], 
                                                     mode='same')
                        input_gradient[batch, channel] += input_grad
        
        if self._optimizer:
            self.weights = self._optimizer.calculate_update(self.weights, self._gradient_weights)
            if self._bias_optimizer:
                self.bias = self._bias_optimizer.calculate_update(self.bias, self._gradient_bias)
        
        return input_gradient
    
    #Weight initialization for convolutional layer

    def initialize(self, weights_initializer, bias_initializer):
        if self.conv_dim == 1: #1D weight shape : (num_kernels, input_channels, kernel_size)
            fan_in = np.prod(self.convolution_shape)
            fan_out = np.prod(self.convolution_shape[1:]) * self.num_kernels
        else: #2D weight shape : (num_kernels, input_channels, kernel_width, kernel_height)
            fan_in = np.prod(self.convolution_shape)
            fan_out = np.prod(self.convolution_shape[1:]) * self.num_kernels
        
        self.weights = weights_initializer.initialize(self.weights.shape, fan_in, fan_out)
        self.bias = bias_initializer.initialize(self.bias.shape, fan_in, fan_out).flatten()