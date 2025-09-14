# Deep Learning Course Implementations

Complete implementation of Deep Learning concepts from NumPy fundamentals to PyTorch

## Course Overview

**Program**: MSc Computational Engineering  
**Focus**: Deep Learning from basics to advanced applications  

## The Journey

### Exercise 0: NumPy Foundations
Built pattern generators and data pipelines from scratch using pure NumPy.

**What I built**:
- Checkerboard, circle, and spectrum pattern generators
- CIFAR-10 image data loader with augmentation
- Custom batch iterator for training data

**Key skills**: Vectorized operations, data preprocessing, memory-efficient coding

### Exercise 1: Neural Networks from Scratch
Implemented complete neural networks without any frameworks.

**What I built**:
- Full neural network class with forward/backward pass
- Dense layers with manual gradient computation
- ReLU and Softmax activations
- Cross-entropy loss and SGD optimizer

**Key skills**: Backpropagation math, gradient computation, optimization algorithms

### Exercise 2: Convolutional Neural Networks
Built CNNs from the ground up with manual convolution operations.

**What I built**:
- 1D/2D convolution layers with stride and padding
- Max pooling operations
- Layer composition for complete CNN architectures

**Key skills**: Convolution mathematics, feature extraction, CNN design

### Exercise 3: Advanced Techniques & RNNs
Implemented modern deep learning techniques and recurrent networks.

**What I built**:
- Batch normalization with moving averages
- Dropout regularization
- Vanilla RNN and LSTM networks
- L1/L2 regularizers

**Results achieved**: **98.83% accuracy** on UCI digit recognition

### Exercise 4: PyTorch Production System
Transitioned to PyTorch for real-world application development.

**What I built**:
- Complete training framework with early stopping
- Custom dataset classes for image classification
- GPU acceleration and model checkpointing
- ONNX export for production deployment

**Application**: Industrial crack detection system

## Key Achievements

🎯 **98.83% accuracy** on UCI hand-written digits  
🔧 **Built everything from scratch** before using frameworks  
🚀 **Production-ready** PyTorch implementation  
📊 **Real-world application** for crack detection  


## Technologies Used

**Core**: Python, NumPy, SciPy  
**Deep Learning**: PyTorch, scikit-learn  
**Visualization**: Matplotlib  
**Production**: ONNX, GPU acceleration

# 🚀 Quick Start

### Prerequisites
```bash
pip install numpy scipy torch matplotlib scikit-learn pandas
```

### Get Started in 3 Steps
```bash
# 1. Clone the repository
git clone https://github.com/Vxrun2000/Deep_Learning.git
cd Deep_Learning

# 2. See what was expected (run unit tests first)
cd exercise0_material/src_to_implement
python NumpyTests.py        # See the requirements

# 3. See my implementation working
python pattern.py           # My pattern generators
python main.py             # My data pipeline
```

### Explore Each Exercise

**🔢 Exercise 0 - NumPy Patterns**
```bash
cd exercise0_material/src_to_implement
python NumpyTests.py        # Unit tests - what was expected
python pattern.py          # My implementation - checkerboard, circles
python main.py             # My CIFAR-10 data pipeline
```

**🧠 Exercise 1 - Neural Networks from Scratch**
```bash
cd exercise1_material/src_to_implement
python NeuralNetworkTests.py   # Unit tests - network requirements
# My implementations are in Layers/ and Optimization/ folders
```

**👁️ Exercise 2 - CNNs**
```bash
cd exercise2_material/src_to_implement
python NeuralNetworkTests.py   # Unit tests - CNN requirements
python SoftConvTests.py        # Convolution validation tests
# My implementations: Conv.py, Pooling.py, Flatten.py
```

**🚀 Exercise 3 - RNN's
```bash
cd exercise3_material/src_to_implement
python NeuralNetworkTests.py   # Unit tests - advanced requirements
python TrainLeNet.py           # My training results (see log.txt)
# My implementations: BatchNormalization.py, RNN.py, LSTM.py
```

**⚡ Exercise 4 - PyTorch Production**
```bash
cd exercise4_material/PYTORCH
python PytorchChallengeTests.py  # Unit tests - production requirements
python train.py      
