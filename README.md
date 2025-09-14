# Deep Learning Course Implementations

A comprehensive implementation of Deep Learning concepts from NumPy fundamentals to PyTorch production systems. This repository contains complete solutions for 5 progressive exercises covering the entire spectrum of neural network development.

## 📋 Course Overview

**Program**: MSc Computational Engineering  
**Focus**: Deep Learning Fundamentals to Advanced Applications  
**Implementation**: From scratch neural networks to production-ready PyTorch models

## 🗂️ Repository Structure

```
Deep-Learning/
├── exercise0_material/     # NumPy Fundamentals & Data Pipeline
├── exercise1_material/     # Neural Networks from Scratch
├── exercise2_material/     # Convolutional Neural Networks
├── exercise3_material/     # Advanced Techniques & RNNs
├── exercise4_material/     # PyTorch Production Challenge
└── README.md
```

## 📚 Exercise Details

### Exercise 0: NumPy Fundamentals & Data Pipeline
**Focus**: Foundation building with NumPy and data handling

**Key Components**:
- **`pattern.py`** - Pattern generation classes using advanced NumPy operations
  - `Checker` class: Creates checkerboard patterns using `np.tile()` and `np.kron()`
  - `Circle` class: Generates circular patterns with meshgrid operations
  - `Spectrum` class: RGB color spectrum generation with linear interpolation
- **`generator.py`** - ImageGenerator for CIFAR-10 data pipeline with augmentation
- **`main.py`** - Testing framework and pattern visualization

**Skills Demonstrated**:
- Vectorized operations using `np.tile()`, `np.kron()`, `np.meshgrid()`
- Data augmentation (rotation, mirroring) and preprocessing
- Memory-efficient array operations and batch processing
- JSON label mapping and CIFAR-10 data organization
- Custom iterator implementation for batch generation

### Exercise 1: Neural Networks from Scratch
**Focus**: Manual implementation of neural network fundamentals without frameworks

**Architecture**:
```
src_to_implement/
├── NeuralNetwork.py       # Core network class with forward/backward pass
├── Layers/
│   ├── FullyConnected.py  # Dense layer with manual gradient computation
│   ├── ReLU.py           # ReLU activation with derivative
│   ├── SoftMax.py        # Softmax output layer
│   └── Base.py           # Abstract base layer interface
└── Optimization/
    ├── Loss.py           # CrossEntropyLoss implementation
    └── Optimizers.py     # SGD with momentum optimizer
```

**Skills Demonstrated**:
- Manual backpropagation implementation from first principles
- Gradient computation using chain rule and matrix calculus
- Custom layer architecture with trainable parameters
- Weight initialization strategies (Xavier, Constant)
- Loss function implementation and optimization algorithms

### Exercise 2: Convolutional Neural Networks
**Focus**: CNN implementation with convolution and pooling operations

**Key Features**:
- **`Conv.py`** - 1D/2D Convolution layer implementation
  - Support for stride and padding configurations
  - Manual convolution using `scipy.signal.correlate`
  - Gradient computation for both weights and inputs
- **`Pooling.py`** - Max pooling with configurable window size
- **`Flatten.py`** - Tensor reshaping between conv and FC layers
- **`Initializers.py`** - Xavier and He weight initialization methods

**Skills Demonstrated**:
- 2D convolution mathematics and implementation
- Feature map computation with stride and padding
- Pooling operations and spatial downsampling
- CNN architecture design and layer composition
- Memory-efficient convolution operations

### Exercise 3: Advanced Techniques & RNNs
**Focus**: Regularization techniques and recurrent neural networks

**Advanced Components**:
- **`BatchNormalization.py`** - Complete batch norm with moving averages
- **`Dropout.py`** - Regularization with training/testing phase handling
- **`RNN.py`** - Vanilla RNN with hidden state management
- **`LSTM.py`** - Long Short-Term Memory with forget/input/output gates
- **`L1_L2_Regularizers.py`** - Weight penalty implementations

**Performance Results**:
- **Best Accuracy**: 98.83% on UCI Hand-written Digits dataset
- **Regularization Impact**: BatchNorm provided best performance
- **Optimizer Comparison**: ADAM vs SGD performance analysis
- **Multiple Datasets**: UCI Digits (98.83%) and Iris (98%) classification

**Detailed Results from Training Log**:
- BatchNorm: 98.83% accuracy (best performance)
- BatchNorm + L2: 97.66-97.99% accuracy
- ADAM optimizer: 95.83-96.66% accuracy  
- L1 regularizer: 96.16-97.16% accuracy
- L2 regularizer: 96.83-97.33% accuracy
- Dropout: 94.82-96.49% accuracy

### Exercise 4: PyTorch Production Challenge
**Focus**: Transition to PyTorch framework with real-world application

**Multi-Component Structure**:
```
exercise4_material/
├── PYTORCH/               # Main PyTorch challenge
│   ├── train.py          # Training pipeline setup
│   ├── trainer.py        # Training loop with early stopping
│   ├── data.py           # ChallengeDataset implementation
│   └── export_onnx.py    # Model deployment preparation
├── FCNN/                 # Neural network reference
└── Numpy/                # NumPy foundation review
```

**Production Components**:
- **`trainer.py`** - Complete training framework with:
  - Early stopping with patience
  - GPU acceleration support
  - F1-score evaluation metrics
  - Checkpoint saving functionality
- **`data.py`** - Custom PyTorch Dataset for image classification
- **`train.py`** - End-to-end training pipeline with data loading

**Application**: Industrial crack detection system
- **Task**: Binary classification for structural crack detection
- **Framework**: PyTorch with GPU acceleration
- **Deployment**: ONNX export for production deployment
- **Evaluation**: F1-score metrics for imbalanced classification

## 🛠️ Technologies Used

- **NumPy** - Fundamental array operations and mathematical computations
- **SciPy** - Signal processing for convolution operations
- **PyTorch** - Deep learning framework for production implementation
- **Matplotlib** - Data visualization and result plotting
- **scikit-learn** - Data preprocessing and evaluation metrics
- **Python 3.x** - Core programming language

## 🏆 Key Achievements

1. **Complete From-Scratch Implementation**: Built neural networks without high-level frameworks
2. **Mathematical Understanding**: Implemented backpropagation and gradient descent manually
3. **Performance Excellence**: Achieved 98.83% accuracy on digit recognition
4. **Production Readiness**: Created deployable PyTorch models with ONNX export
5. **Real-World Application**: Developed crack detection system for industrial use
6. **Comprehensive Testing**: All implementations validated against reference solutions

## 📊 Performance Metrics

### Exercise 3 - Detailed Results

| Technique | UCI Digits Accuracy | Iris Dataset Accuracy |
|-----------|--------------------|-----------------------|
| **BatchNorm** | **98.83%** | **98%** |
| BatchNorm + L2 | 97.66-97.99% | - |
| ADAM Optimizer | 95.83-96.66% | - |
| L1 Regularizer | 96.16-97.16% | - |
| L2 Regularizer | 96.83-97.33% | - |
| Dropout | 94.82-96.49% | 96% |

### Overall Exercise Summary

| Exercise | Task | Performance | Framework |
|----------|------|-------------|-----------|
| 0 | Pattern Generation & Data Pipeline | ✅ Pass | NumPy |
| 1 | Neural Network Implementation | ✅ Pass | From Scratch |
| 2 | CNN Development | ✅ Pass | From Scratch |
| 3 | Advanced Techniques & RNNs | **98.83% Accuracy** | From Scratch |
| 4 | PyTorch Production System | ✅ Production Ready | PyTorch |

## 🚀 Getting Started

### Prerequisites
```bash
numpy>=1.19.0
scipy>=1.7.0
torch>=1.9.0
torchvision>=0.10.0
matplotlib>=3.3.0
scikit-learn>=0.24.0
scikit-image>=0.18.0
pandas>=1.3.0
tqdm>=4.60.0
```

### Installation
```bash
git clone https://github.com/Vxrun2000/Deep_Learning.git
cd Deep_Learning
pip install -r requirements.txt
```

### Running Exercises

#### Exercise 0 - NumPy Fundamentals
```bash
cd exercise0_material/src_to_implement
python pattern.py          # Generate and visualize patterns
python main.py             # Run ImageGenerator tests
python NumpyTests.py        # Validate implementations
```

#### Exercise 1 - Neural Networks from Scratch
```bash
cd exercise1_material/src_to_implement
python NeuralNetworkTests.py   # Run all network tests
```

#### Exercise 2 - Convolutional Neural Networks
```bash
cd exercise2_material/src_to_implement
python NeuralNetworkTests.py   # Test CNN implementations
python SoftConvTests.py        # Validate convolution operations
```

#### Exercise 3 - Advanced Techniques & RNNs
```bash
cd exercise3_material/src_to_implement
python TrainLeNet.py           # Train LeNet with different techniques
python NeuralNetworkTests.py   # Test all advanced components
```

#### Exercise 4 - PyTorch Production Challenge
```bash
cd exercise4_material/PYTORCH
python train.py               # Train crack detection model
python export_onnx.py         # Export model for deployment
```

## 📈 Learning Progression

This repository demonstrates a complete learning journey:

1. **Foundation** → NumPy mastery and data handling
2. **Core Concepts** → Manual neural network implementation  
3. **Advanced Architecture** → CNN development and optimization
4. **Cutting-Edge Techniques** → RNNs, regularization, and modern practices
5. **Production Deployment** → PyTorch framework and real-world applications

## 🔍 Code Quality Features

- **Modular Design**: Clean separation of concerns across layers
- **Comprehensive Testing**: Validation scripts for each component
- **Documentation**: Detailed comments and docstrings throughout
- **Performance Optimization**: Vectorized operations and efficient algorithms
- **Industry Standards**: Following PyTorch conventions and best practices
- **Error Handling**: Robust input validation and error management

## 🧠 Technical Highlights

### Mathematical Implementations
- **Manual Backpropagation**: Complete gradient computation from scratch
- **Convolution Operations**: 1D/2D convolution with stride and padding
- **Batch Normalization**: Moving averages and gradient computation
- **RNN/LSTM**: Hidden state management and gradient flow

### Production Features
- **Early Stopping**: Preventing overfitting with patience mechanism
- **Checkpoint Saving**: Model state preservation during training
- **GPU Acceleration**: CUDA support for faster computation
- **ONNX Export**: Model deployment for production environments

## 📝 Dataset Information

- **CIFAR-10**: 60,000 32x32 color images in 10 classes (Exercise 0)
- **UCI Hand-written Digits**: 1,797 8x8 grayscale digit images (Exercise 3)
- **Iris Dataset**: 150 samples with 4 features for classification (Exercise 3)
- **Industrial Crack Detection**: 2000+ labeled crack images (Exercise 4)

## 🤝 Contact

**Author**: Varun Narendra Kumar  
**GitHub**: [@Vxrun2000](https://github.com/Vxrun2000)  
**Repository**: [Deep_Learning](https://github.com/Vxrun2000/Deep_Learning)

## 📄 License

This project is part of academic coursework and is shared for educational purposes.

---

*This repository showcases a complete deep learning education journey from mathematical foundations to production-ready implementations, demonstrating both theoretical understanding and practical application skills.*