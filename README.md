# Neural Network for MNIST Classification

> An educational implementation of a neural network from scratch, built to understand the fundamentals of deep learning algorithms.

A Python project developed during the Neural Network and Deep Learning course, featuring a complete implementation of a feedforward neural network **without using deep learning frameworks**. The network is trained and evaluated on the **MNIST dataset**, a standard benchmark for handwritten digit classification.

---

## Features

### Core Implementation
- **From-Scratch Neural Network**: Fully custom implementation using only NumPy
- **Flexible Architecture**: Support for 1-3 hidden layers with configurable sizes
- **Multiple Activation Functions**: Sigmoid, Tanh, ReLU, and Identity
- **Backpropagation Algorithm**: Manual gradient computation and weight updates
- **Cross-Entropy Loss**: Optimized for multi-class classification with softmax

### Training & Validation
- **Grid Search**: Automated hyperparameter exploration across architectures, learning rates, and activations
- **Holdout Validation**: Fast initial screening of configuration combinations
- **K-Fold Cross-Validation**: Robust performance evaluation of top configurations
- **Early Stopping**: Configurable patience to prevent overfitting
- **Incremental Saving**: Results saved after each experiment (safe for interruptions)

### Results Management
- **Multi-Format Export**: JSON (structured), CSV (tabular), and Markdown (readable)
- **Comprehensive Metrics**: Training/validation/test accuracy, loss, epochs trained, and timing
- **Statistical Summaries**: Automatic performance statistics and ranking
- **Full Reproducibility**: All hyperparameters and configurations logged

### Visualization
- **Prediction Display**: View model predictions on test samples
- **Comparison View**: Side-by-side display of true vs predicted labels with images

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Peppebalzanoo/neural-network-MNIST-2025.git
cd neural-network-MNIST-2025

# Install dependencies
pip install -r requirements.txt
```

### Run Examples

Explore ready-to-use code examples in `examples.py`:

```bash
# Run a specific example (1-7)
python examples.py 1

# Run all examples sequentially
python examples.py all
```

### Run Full Experiments

Execute the complete hyperparameter search pipeline:

```bash
python main.py
```

This automatically:
1. Loads and preprocesses the MNIST dataset
2. Runs holdout validation on all configuration combinations
3. Performs k-fold cross-validation on top 3 configurations
4. Saves results to `experiments_results.json`, `.csv`, and `RESULTS.md`

---

## Usage

### Training a Single Network

Three approaches with increasing sophistication:

#### Option 1: Direct Training

Manual control over data splitting and training:

```python
from network import Network
from loader import DataLoader
from error import cross_entropy
import activation as act

# Load and split data
loader = DataLoader("./MNIST")
X_train, Y_train, X_test, Y_test = loader.get_train_test_data()
X_train_split, Y_train_split, X_valid, Y_valid = loader.split_data(0.8)

# Create and train network
network = Network(784, [128], 10, activation_function=act.relu)
history = network.fit(
    X_train_split, Y_train_split, X_valid, Y_valid,
    error_function=cross_entropy,
    epoch_number=100, eta=0.01, patience=10
)

# Evaluate
Z_test = network.forward_propagation(X_test)
accuracy = network.get_accuracy(Z_test, Y_test)
print(f"Test Accuracy: {accuracy:.2%}")
```

#### Option 2: Holdout Validation

Let the `Trainer` handle data splitting (recommended for quick experiments):

```python
from network import Network
from loader import DataLoader
from trainer import Trainer
from error import cross_entropy
import activation as act

loader = DataLoader("./MNIST")
X_train, Y_train, X_test, Y_test = loader.get_train_test_data()

network = Network(784, [128], 10, activation_function=act.relu)
history = Trainer.holdout_validation(
    loader, network, cross_entropy,
    epoch_number=100, eta=0.01, patience=10, train_ratio=0.8
)

Z_test = network.forward_propagation(X_test)
accuracy = network.get_accuracy(Z_test, Y_test)
print(f"Test Accuracy: {accuracy:.2%}")
```

#### Option 3: K-Fold Cross-Validation

Most robust evaluation (slower, recommended for final model selection):

```python
from loader import DataLoader
from trainer import Trainer
from error import cross_entropy
import activation as act

loader = DataLoader("./MNIST")
X_train, Y_train, X_test, Y_test = loader.get_train_test_data()

best_model, avg_history = Trainer.kfold_cross_validation(
    loader, 784, [128], 10, cross_entropy,
    k=5, epoch_number=100, eta=0.01, patience=10,
    activation_function=act.relu
)

Z_test = best_model.forward_propagation(X_test)
accuracy = best_model.get_accuracy(Z_test, Y_test)
print(f"Test Accuracy: {accuracy:.2%}")
```

---

### Customizing Experiments

Configure hyperparameters in `config.py`:

```python
# Architecture exploration
HIDDEN_CONFIGURATIONS = [
    [50],          # Single layer
    [100, 50],     # Two layers
    [128, 64, 32], # Three layers
]

# Learning rate search
LEARNING_RATES = [0.1, 0.5, 1.0]

# Activation function comparison
ACTIVATION_FUNCTIONS = [
    (act.sigmoid, "sigmoid"),
    (act.tanh, "tanh"),
    (act.relu, "relu"),
]

# Training parameters
EPOCH_NUMBER = 1000
PATIENCE = 10
K_FOLDS = 5
TOP_N_FOR_KFOLD = 3
```

---

### Advanced Examples

#### Custom Experiment

Run a specific configuration with custom parameters:

```python
from experiments import ExperimentRunner
from loader import DataLoader
from error import cross_entropy
import activation as act

loader = DataLoader("./MNIST")
X_train, Y_train, X_test, Y_test = loader.get_train_test_data()
runner = ExperimentRunner(loader, X_test, Y_test)

result = runner.run_single_experiment(
    input_neurons=784,
    hidden_neurons=[100, 50],
    output_neurons=10,
    eta=0.1, epoch_number=500, patience=10,
    error_function=cross_entropy,
    activation_function=act.tanh,
    activation_name="tanh",
    use_kfold=False
)

print(f"Test Accuracy: {result['test_accuracy']:.2%}")
```

#### Visualize Predictions

Display model predictions on test samples:

```python
from visualization import Visualizer
from network import Network
from loader import DataLoader
import activation as act

loader = DataLoader("./MNIST")
X_train, Y_train, X_test, Y_test = loader.get_train_test_data()

# Train your network
network = Network(784, [128], 10, activation_function=act.relu)
# ... training code ...

# Show predictions
Visualizer.show_multiple_predictions(network, X_test, Y_test, num_samples=5)
```

---

## Project Structure

```
neural-network-MNIST-2025/
├── main.py                 # Main experiment pipeline
├── examples.py             # Ready-to-run usage examples
├── config.py               # Hyperparameter configuration
├── network.py              # Neural network core implementation
├── trainer.py              # Training strategies (holdout, k-fold)
├── experiments.py          # Grid search and experiment management
├── loader.py               # MNIST data loading and preprocessing
├── activation.py           # Activation functions
├── error.py                # Loss functions
├── results_manager.py      # Results export (JSON, CSV, Markdown)
├── visualization.py        # Prediction visualization
├── requirements.txt        # Python dependencies
└── MNIST/                  # Dataset directory
    ├── mnist_train.csv
    └── mnist_test.csv
```

---

## Experiment Results

Complete results from the latest experimental run (2025-11-13):

> **Note:** Additional experiments with different architectures are in progress and will be added to this section.

### Top Performing Configurations

| Rank | Architecture | Activation | η (LR) | Test Acc | Valid Acc | Epochs | Valid Error |
|------|--------------|------------|--------|----------|-----------|--------|-------------|
| 1 | [100] | ReLU | 0.5 | **96.51%** | 96.38% | 1000 | 0.1244 |
| 2 | [50] | ReLU | 0.5 | **96.34%** | 95.62% | 1000 | 0.1581 |
| 3 | [100] | Tanh | 0.5 | **96.09%** | 95.86% | 1000 | 0.1474 |
| 4 | [50] | Tanh | 0.5 | **96.00%** | 95.49% | 1000 | 0.1507 |
| 5 | [100] | Sigmoid | 1.0 | 11.35% | 11.22% | 11 | 2.3469 |

### Full Results Summary

#### Architecture: [50] Hidden Neurons

| Activation | η | Test Acc | Valid Acc | Epochs | Initial Error | Final Error | Status |
|------------|---|----------|-----------|--------|---------------|-------------|--------|
| Sigmoid | 0.1 | 11.35% | 10.81% | 19 | 2.3026 | 2.3019 | Failed |
| Tanh | 0.1 | 92.02% | 91.58% | 1000 | 2.3026 | 0.2976 | Success |
| ReLU | 0.1 | 91.52% | 91.03% | 1000 | 2.3026 | 0.3087 | Success |
| Sigmoid | 0.5 | 92.75% | 92.38% | 1000 | 2.3026 | 0.2592 | Success |
| **Tanh** | **0.5** | **96.00%** | **95.49%** | **1000** | **2.3026** | **0.1507** | Success |
| **ReLU** | **0.5** | **96.34%** | **95.62%** | **1000** | **2.3026** | **0.1581** | Success |
| Sigmoid | 1.0 | 94.50% | 94.07% | 1000 | 2.3026 | 0.2107 | Success |
| Tanh | 1.0 | 92.13% | 91.82% | 145 | 2.3026 | 0.2733 | Early Stop |
| ReLU | 1.0 | 39.75% | 40.21% | 27 | 2.3026 | 2.0620 | Unstable |

#### Architecture: [100] Hidden Neurons

| Activation | η | Test Acc | Valid Acc | Epochs | Initial Error | Final Error | Status |
|------------|---|----------|-----------|--------|---------------|-------------|--------|
| Sigmoid | 0.1 | 11.35% | 10.93% | 14 | 2.3025 | 2.3018 | Failed |
| Tanh | 0.1 | 91.97% | 91.41% | 1000 | 2.3026 | 0.2951 | Success |
| ReLU | 0.1 | 91.64% | 91.18% | 1000 | 2.3026 | 0.3161 | Success |
| Sigmoid | 0.5 | 92.61% | 92.29% | 1000 | 2.3026 | 0.2709 | Success |
| **Tanh** | **0.5** | **96.09%** | **95.86%** | **1000** | **2.3026** | **0.1474** | Success |
| **ReLU** | **0.5** | **96.51%** | **96.38%** | **1000** | **2.3026** | **0.1244** | Success |
| Sigmoid | 1.0 | 11.35% | 11.22% | 11 | 2.3026 | 2.3469 | Failed |
| Tanh | 1.0 | 33.89% | 33.73% | 26 | 2.3026 | 2.9349 | Unstable |
| ReLU | 1.0 | 39.71% | 39.53% | 26 | 2.3026 | 1.6202 | Unstable |

### Key Insights

**Best Overall Performance:**
- **Configuration:** 784 × [100] × 10 with ReLU, η=0.5
- **Test Accuracy:** 96.51%
- **Validation Accuracy:** 96.38%
- **Final Validation Error:** 0.1244

**Observations:**
- **Optimal Learning Rate:** η=0.5 consistently produces best results
- **Architecture Impact:** [100] neurons slightly outperforms [50] neurons
- **Activation Functions:** ReLU and Tanh perform equally well at η=0.5
- **Training Stability:** 
  - η=0.1 is too low for Sigmoid (fails to learn)
  - η=1.0 causes instability with ReLU (39-40% accuracy)
  - η=0.5 provides optimal balance

**Failed Configurations:**
- Sigmoid with η=0.1 and η=1.0: Network predicts only class 1 (11.35% = baseline)
- ReLU/Tanh with η=1.0: Gradient explosion causing poor convergence

---

## Output Files

Generated after running experiments:

- **`experiments_results.json`**: Structured results with full details
- **`experiments_results.csv`**: Tabular format for analysis
- **`RESULTS.md`**: Human-readable markdown report

---

## Requirements

- **Python**: 3.7+
- **NumPy**: Numerical computations
- **Matplotlib**: Visualization
- **Pandas**: CSV export

See `requirements.txt` for specific versions.

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.