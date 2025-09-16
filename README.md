# Neural Network And Deep Learning

### The project is certainly not the best implementation possible, but it aims to understand the fundamentals of neural networks.

An educational Python project developed during the Neural Network And Deep Learning course,
this repository features the implementation of a neural network from scratch, without external libraries, and includes fundamental neural network algorithms.
The network was trained and evaluated on the **MNIST dataset**, a benchmark dataset of handwritten digits commonly used for image classification tasks.

---

## Network Architecture

- **Layers:** 784 × [50] × 10  
- **Weights Shapes:** [(50, 784), (10, 50)]  
- **Biases Shapes:** [(50, 1), (10, 1)]  
- **Activation Functions:** ['tanh', 'identity']  

---

## Training & Evaluation Results

- **Learning Rate:** 0.1  
- **Test Set Accuracy (Holdout-Val):** 0.9216  
- **Test Set Accuracy (K-Fold):** 0.9191  
