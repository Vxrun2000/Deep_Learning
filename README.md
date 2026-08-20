# Deep Learning from Scratch

Implementations of core deep-learning concepts developed during my **M.Sc. Computational Engineering** studies.

This repository follows a progression from fundamental numerical operations in **NumPy** to manually implemented neural networks, convolutional and recurrent architectures, and finally modular **PyTorch** training workflows.

The main objective is to understand the mathematics and computational mechanisms behind deep-learning models before relying on high-level frameworks.

## Learning Progression

```text
NumPy fundamentals
        ↓
Neural networks + backpropagation
        ↓
Convolutional neural networks
        ↓
Regularization + RNN + LSTM
        ↓
PyTorch training workflows
```

## Academic Context

This repository contains implementations developed as part of graduate Deep Learning coursework in the **M.Sc. Computational Engineering** program.

The exercise framework, unit tests, datasets, and selected interfaces were provided as part of the course. My work focused on implementing the required numerical operations, neural-network components, training procedures, and PyTorch workflows within that framework.

The repository is preserved as a technical record of the progression from first-principles implementation to framework-based deep learning.

## Implementations

### 1. NumPy Foundations

Fundamental numerical operations used throughout neural-network implementations.

Topics include:

* NumPy array operations
* Vectorization
* Data preprocessing
* Batch generation
* Numerical manipulation of multidimensional data

### 2. Neural Networks from Scratch

Implementation of the core components required to train fully connected neural networks.

Topics include:

* Dense layers
* Forward propagation
* Backpropagation
* Manual gradient computation
* ReLU activation
* Softmax activation
* Cross-entropy loss
* Stochastic gradient descent
* Parameter updates

This section focuses on translating the mathematical formulation of neural networks directly into code.

### 3. Convolutional Neural Networks

Implementation of convolutional operations and CNN building blocks.

Topics include:

* 1D and 2D convolution
* Stride
* Padding
* Pooling
* Flattening
* Forward and backward passes through convolutional layers

The emphasis is on understanding how spatial feature extraction is implemented computationally.

### 4. Advanced Neural-Network Components

Implementation of additional architectures and regularization techniques.

Topics include:

* Batch normalization
* Dropout
* L1 and L2 regularization
* Recurrent neural networks
* Long Short-Term Memory networks
* Sequential-data processing

### 5. PyTorch Training Pipeline

The final stage transitions from first-principles implementations to framework-based deep learning using PyTorch.

Topics include:

* Custom datasets and dataloaders
* Modular model definitions
* Training and validation loops
* GPU acceleration
* Model checkpointing
* Early stopping
* Evaluation workflows
* ONNX model export for portable inference

## Example Result

An example handwritten-digit classification workflow achieved approximately:

**98.83% classification accuracy**

on the corresponding digit-recognition dataset used in the coursework.

The purpose of the repository is primarily to demonstrate implementation depth and understanding of deep-learning fundamentals rather than benchmark optimization.

## Technical Coverage

| Area                     | Concepts                                                   |
| ------------------------ | ---------------------------------------------------------- |
| Numerical Computing      | NumPy, vectorized operations, multidimensional arrays      |
| Neural Networks          | Forward propagation, backpropagation, gradient computation |
| Optimization             | SGD, parameter updates, regularization                     |
| CNNs                     | Convolution, padding, stride, pooling                      |
| Recurrent Models         | RNN, LSTM                                                  |
| Regularization           | Dropout, BatchNorm, L1/L2                                  |
| Deep Learning Frameworks | PyTorch                                                    |
| Training Infrastructure  | Checkpointing, early stopping, GPU support                 |
| Model Portability        | ONNX export                                                |

## Repository Structure

```text
deep-learning-from-scratch/
├── exercise0_material/    # NumPy foundations
├── exercise1_material/    # Neural-network fundamentals
├── exercise2_material/    # Convolutional neural networks
├── exercise3_material/    # RNN, LSTM and regularization
├── exercise4_material/    # PyTorch workflows
└── README.md
```

Individual exercise directories contain the corresponding implementations, tests, utilities, and training scripts.

## Technologies

* Python
* NumPy
* SciPy
* PyTorch
* pandas
* Matplotlib
* scikit-learn
* ONNX

## Setup

Clone the repository:

```bash
git clone https://github.com/Vxrun2000/deep-learning-from-scratch.git
cd deep-learning-from-scratch
```

Install the required Python packages according to the dependencies used by the individual exercises.

A typical environment includes:

```bash
pip install numpy scipy pandas matplotlib scikit-learn torch onnx
```

## Purpose

This repository demonstrates my progression from implementing neural-network components directly from their mathematical formulation to building complete training workflows with PyTorch.

The focus is on **understanding, implementation, and scientific computing**, rather than solely applying pre-built deep-learning APIs.
