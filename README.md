# Neural Network for MNIST Classification

> An educational implementation of a neural network from scratch, built to understand the fundamentals of deep learning algorithms.

A Python project developed during the Neural Network and Deep Learning course, featuring a complete implementation of a feedforward neural network **without using deep learning frameworks**. The network is trained and evaluated on the **MNIST dataset**, a standard benchmark for handwritten digit classification.

---

## Features

### Core Implementation
- **From-Scratch Neural Network**: Fully custom implementation using only NumPy
- **Flexible Architecture**: Support for multiple hidden layers with configurable sizes
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

Explore ready-to-use code examples in `examples.py`. Open the file and uncomment the example you want to run:
```python
# Edit examples.py and uncomment one or more examples:
if __name__ == "__main__":
    example_direct_training()              # Option 1: Direct training
    # example_holdout_validation()         # Option 2: Holdout validation
    # example_kfold_cross_validation()     # Option 3: K-fold CV
    # example_custom_experiment()          # Custom experiment with ExperimentRunner
    # example_visualize_predictions()      # Visualize predictions
    # example_compare_activations()        # Compare activation functions
    # example_compare_architectures()      # Compare architectures
```

Then run:
```bash
python examples.py
```

**Available Examples:**
- `example_direct_training()` - Manual control over training
- `example_holdout_validation()` - Quick training with Trainer
- `example_kfold_cross_validation()` - Robust evaluation with K-fold
- `example_custom_experiment()` - Custom experiment with automatic result saving
- `example_visualize_predictions()` - Visual predictions on test samples
- `example_compare_activations()` - Compare sigmoid, tanh, and relu
- `example_compare_architectures()` - Compare different network sizes

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

Three approaches with increasing sophistication:

### Which Method Should I Use?

| Your Goal | Method | Speed | Reliability |
|-----------|--------|-------|-------------|
| Quick experiments | Holdout Validation | Fast | Medium |
| Final evaluation | K-Fold CV | Slow | High |
| Custom training logic | Manual Control | Fast | Depends |

---

### Option 1: Holdout Validation (Recommended)

**Let the `Trainer` handle data splitting (recommended for quick experiments)**
```python
from network import Network
from loader import DataLoader
from trainer import Trainer
from error import cross_entropy
from visualization import Visualizer
from results_manager import ResultsManager
import activation as act

# Load data
loader = DataLoader("./MNIST")
X_train, Y_train, X_test, Y_test = loader.get_train_test_data()

# Create and train network
network = Network(784, [128], 10, activation_function=act.relu)
history = Trainer.holdout_validation(
    loader, network, cross_entropy,
    epoch_number=100, eta=0.01, patience=10, train_ratio=0.8
)

# Evaluate on test set
Z_test = network.forward_propagation(X_test)
test_accuracy = network.get_accuracy(Z_test, Y_test)
print(f"Test Accuracy: {test_accuracy:.2%}")

# Save results (optional)
result = {
    "hidden_neurons": [128],
    "eta": 0.01,
    "activation": "relu",
    "epoch_number": 100,
    "patience": 10,
    "method": "holdout",
    "test_accuracy": float(test_accuracy),
    "epochs_trained": history["epochs_trained"],
    "initial_valid_error": history["initial_valid_error"],
    "initial_valid_accuracy": history["initial_valid_accuracy"],
    "final_valid_error": history["final_valid_error"],
    "final_valid_accuracy": history["final_valid_accuracy"],
    "best_valid_error": history["best_valid_error"],
    "status": "success"
}
ResultsManager.append_result(result, "my_experiments.json")

# Visualize predictions (optional)
Visualizer.show_multiple_predictions(network, X_test, Y_test, num_samples=5)
```

**Use when:** Learning, prototyping, quick experiments  
**Limitation:** Single validation split  
**Note:** Results are NOT saved automatically. Use `ResultsManager.append_result()` to save.

---

### Option 2: K-Fold Cross-Validation

**Most robust evaluation (slower, recommended for final model selection)**
```python
from loader import DataLoader
from trainer import Trainer
from error import cross_entropy
from visualization import Visualizer
from results_manager import ResultsManager
import activation as act

# Load data
loader = DataLoader("./MNIST")
X_train, Y_train, X_test, Y_test = loader.get_train_test_data()

# Train with K-fold cross-validation
best_model, avg_history = Trainer.kfold_cross_validation(
    loader, 784, [128], 10, cross_entropy,
    k=5, epoch_number=200, eta=0.01, patience=10,
    activation_function=act.relu
)

# Evaluate best model
Z_test = best_model.forward_propagation(X_test)
test_accuracy = best_model.get_accuracy(Z_test, Y_test)
print(f"Test Accuracy: {test_accuracy:.2%}")
print(f"Avg Validation Accuracy (K-fold): {avg_history['final_valid_accuracy']:.2%}")

# Save results (optional)
result = {
    "hidden_neurons": [128],
    "eta": 0.01,
    "activation": "relu",
    "epoch_number": 200,
    "patience": 10,
    "method": "k-fold",
    "k_folds": 5,
    "test_accuracy": float(test_accuracy),
    "epochs_trained": int(avg_history["epochs_trained"]),
    "initial_valid_error": avg_history["initial_valid_error"],
    "initial_valid_accuracy": avg_history["initial_valid_accuracy"],
    "final_valid_error": avg_history["final_valid_error"],
    "final_valid_accuracy": avg_history["final_valid_accuracy"],
    "best_valid_error": avg_history["best_valid_error"],
    "status": "success"
}
ResultsManager.append_result(result, "my_experiments.json")

# Visualize predictions (optional)
Visualizer.show_multiple_predictions(best_model, X_test, Y_test, num_samples=10)
```

**Use when:** Thesis results, final evaluation, comparing architectures  
**Note:** Trains K models (5x slower than holdout). Results are NOT saved automatically.

---

### Option 3: Manual Control (Advanced)

**Manual control over data splitting and training**
```python
from network import Network
from loader import DataLoader
from error import cross_entropy
from visualization import Visualizer
from results_manager import ResultsManager
import activation as act

# Load and manually split data
loader = DataLoader("./MNIST")
X_train, Y_train, X_test, Y_test = loader.get_train_test_data()
X_train_split, Y_train_split, X_valid, Y_valid = loader.split_data(0.8)

# Create network
network = Network(784, [128], 10, activation_function=act.relu)

# Train with full control
history = network.fit(
    X_train_split, Y_train_split, X_valid, Y_valid,
    error_function=cross_entropy,
    epoch_number=100, eta=0.01, patience=10
)

# Evaluate
Z_test = network.forward_propagation(X_test)
test_accuracy = network.get_accuracy(Z_test, Y_test)
print(f"Test Accuracy: {test_accuracy:.2%}")
print(f"Epochs trained: {history['epochs_trained']}")
print(f"Final validation error: {history['final_valid_error']:.4f}")

# Save results (optional)
result = {
    "hidden_neurons": [128],
    "eta": 0.01,
    "activation": "relu",
    "epoch_number": 100,
    "patience": 10,
    "method": "manual",
    "test_accuracy": float(test_accuracy),
    "epochs_trained": history["epochs_trained"],
    "initial_valid_error": history["initial_valid_error"],
    "initial_valid_accuracy": history["initial_valid_accuracy"],
    "final_valid_error": history["final_valid_error"],
    "final_valid_accuracy": history["final_valid_accuracy"],
    "best_valid_error": history["best_valid_error"],
    "status": "success"
}
ResultsManager.append_result(result, "my_experiments.json")

# Visualize predictions (optional)
Visualizer.show_prediction(network, X_test, Y_test, random_seed=42)
```

**Use when:** Custom data splits, data augmentation, multi-stage training, debugging  
**Note:** Results are NOT saved automatically. Use `ResultsManager.append_result()` to save.

**Common Use Cases:**
```python
# 1. Custom data split (first 5000 samples only)
X_train_custom = X_train[:5000]
Y_train_custom = Y_train[:5000]
network.fit(X_train_custom, Y_train_custom, X_valid, Y_valid,
           cross_entropy, epoch_number=100, eta=0.01, patience=10)

# 2. Multi-stage training with different learning rates
network.fit(X_train, Y_train, X_valid, Y_valid,
           cross_entropy, epoch_number=50, eta=0.1, patience=5)  # Stage 1: high LR
network.fit(X_train, Y_train, X_valid, Y_valid,
           cross_entropy, epoch_number=50, eta=0.01, patience=5) # Stage 2: low LR

# 3. Debugging with small subset
X_small = X_train[:1000]
Y_small = Y_train[:1000]
network.fit(X_small, Y_small, X_valid[:200], Y_valid[:200],
           cross_entropy, epoch_number=10, eta=0.01, patience=3)
```

---

### Saving Results

Results are saved in JSON format by default, with optional CSV and Markdown export.

#### Save Single Experiment
```python
from results_manager import ResultsManager

# After training, create result dictionary
result = {
    "hidden_neurons": [128],
    "eta": 0.01,
    "activation": "relu",
    "test_accuracy": 0.9651,
    # ... other fields
}

# Save (appends to existing file or creates new)
ResultsManager.append_result(result, "my_experiments.json")
```

#### Save Multiple Experiments
```python
from results_manager import ResultsManager

# Collect results in a list
results = []

# Experiment 1
result1 = {...}  # Your result dictionary
results.append(result1)

# Experiment 2
result2 = {...}  # Your result dictionary
results.append(result2)

# Save all at once
ResultsManager.save_json(results, "comparison.json")
ResultsManager.save_csv(results, "comparison.csv")
ResultsManager.generate_markdown_report(results, "COMPARISON.md")
```

#### Automatic Saving

When using `ExperimentRunner` or `main.py`, results are saved automatically:
```python
from experiments import ExperimentRunner
from loader import DataLoader
from error import cross_entropy
import activation as act

loader = DataLoader("./MNIST")
X_train, Y_train, X_test, Y_test = loader.get_train_test_data()
runner = ExperimentRunner(loader, X_test, Y_test)

# Results are automatically saved to experiments_results.json
result = runner.run_single_experiment(
    input_neurons=784, hidden_neurons=[128], output_neurons=10,
    eta=0.1, epoch_number=100, patience=10,
    error_function=cross_entropy,
    activation_function=act.relu, activation_name="relu",
    use_kfold=False
)
```

---

### Visualizing Results

After training, you can visualize predictions:
```python
from visualization import Visualizer

# Show a single prediction with specific seed
Visualizer.show_prediction(network, X_test, Y_test, random_seed=42)

# Show multiple predictions in a grid
Visualizer.show_multiple_predictions(network, X_test, Y_test, num_samples=10)
```

To view experiment results after running `main.py`:
```bash
# View JSON results
cat experiments_results.json

# View CSV results (can open in Excel/LibreOffice)
cat experiments_results.csv

# View Markdown report (human-readable)
cat RESULTS.md
```

Or load results programmatically:
```python
import json

# Load experiment results
with open('experiments_results.json', 'r') as f:
    data = json.load(f)

# Access experiments
experiments = data['experiments']

# Find best configuration
best = max(experiments, key=lambda x: x['test_accuracy'])

print(f"Best configuration:")
print(f"  Architecture: {best['hidden_neurons']}")
print(f"  Activation: {best['activation']}")
print(f"  Learning rate: {best['eta']}")
print(f"  Test accuracy: {best['test_accuracy']:.2%}")
```

---

### Typical Workflow
```python
# 1. Quick test with Holdout
network = Network(784, [128], 10, activation_function=act.relu)
history = Trainer.holdout_validation(loader, network, cross_entropy,
                                    epoch_number=50, eta=0.01, patience=5)

# 2. Grid search to find best hyperparameters
# Edit config.py, then run:
# python main.py

# 3. Final validation with K-Fold using best hyperparameters
best_model, avg_history = Trainer.kfold_cross_validation(
    loader, 784, [128], 10, cross_entropy,
    k=5, epoch_number=200, eta=0.1, patience=10,  # Use best eta from step 2
    activation_function=act.relu
)

# 4. Save final results
result = {
    "hidden_neurons": [128],
    "eta": 0.1,
    "activation": "relu",
    "test_accuracy": float(best_model.get_accuracy(Z_test, Y_test)),
    # ... other metrics
}
ResultsManager.append_result(result, "final_results.json")
```

---

## Customizing Experiments

Configure hyperparameters in `config.py` to run custom experiments with `main.py`:

### Example 1: Testing Different Architectures
```python
# In config.py, modify:
HIDDEN_CONFIGURATIONS = [
    [50],          # Single layer with 50 neurons
    [100],         # Single layer with 100 neurons
    [128],         # Single layer with 128 neurons
    [100, 50],     # Two layers: 100 -> 50
    [128, 64],     # Two layers: 128 -> 64
    [128, 64, 32], # Three layers: 128 -> 64 -> 32
]
```

Then run:
```bash
python main.py
```

This will test all 6 architectures with all combinations of learning rates and activation functions.

### Example 2: Fine-Tuning Learning Rates
```python
# In config.py, modify:
LEARNING_RATES = [0.01, 0.05, 0.1, 0.5, 1.0]  # Test 5 different learning rates

# Use only one architecture to speed up
HIDDEN_CONFIGURATIONS = [[100]]

# Use only best activation from previous experiments
ACTIVATION_FUNCTIONS = [
    (act.relu, "relu"),
]
```

**Result:** Tests 5 learning rates × 1 architecture × 1 activation = 5 experiments

### Example 3: Comparing Activation Functions
```python
# In config.py, modify:
ACTIVATION_FUNCTIONS = [
    (act.sigmoid, "sigmoid"),
    (act.tanh, "tanh"),
    (act.relu, "relu"),
    (act.identity, "identity"),  # Add identity for comparison
]

# Use fixed architecture and learning rate
HIDDEN_CONFIGURATIONS = [[128]]
LEARNING_RATES = [0.1]
```

**Result:** Tests 4 activation functions on the same architecture

### Example 4: Long Training Experiments
```python
# In config.py, modify:
EPOCH_NUMBER = 2000      # Train longer
PATIENCE = 20            # More patience for early stopping
K_FOLDS = 10             # More folds for robust evaluation
TOP_N_FOR_KFOLD = 5      # Evaluate top 5 configurations with k-fold
```

### Example 5: Quick Testing Setup
```python
# In config.py, modify for fast iteration:
HIDDEN_CONFIGURATIONS = [[50]]      # Single small architecture
LEARNING_RATES = [0.1]              # Single learning rate
ACTIVATION_FUNCTIONS = [(act.relu, "relu")]  # Single activation
EPOCH_NUMBER = 50                   # Few epochs
PATIENCE = 5                        # Early stopping
TOP_N_FOR_KFOLD = 1                # Only best config to k-fold
```

---

## Advanced Examples

### Example 1: Single Custom Experiment

Run a specific configuration and save results:
```python
from experiments import ExperimentRunner
from loader import DataLoader
from error import cross_entropy
import activation as act

# Setup
loader = DataLoader("./MNIST")
X_train, Y_train, X_test, Y_test = loader.get_train_test_data()
runner = ExperimentRunner(loader, X_test, Y_test)

# Run single experiment (automatically saved to experiments_results.json)
result = runner.run_single_experiment(
    input_neurons=784,
    hidden_neurons=[100, 50],      # Two hidden layers
    output_neurons=10,
    eta=0.1,                       # Learning rate
    epoch_number=500,
    patience=10,
    error_function=cross_entropy,
    activation_function=act.tanh,
    activation_name="tanh",
    use_kfold=False                # Use holdout instead of k-fold
)

# Print results
print(f"Test Accuracy: {result['test_accuracy']:.2%}")
print(f"Validation Accuracy: {result['final_valid_accuracy']:.2%}")
print(f"Epochs Trained: {result['epochs_trained']}")
print("✓ Results automatically saved to experiments_results.json")
```

### Example 2: Compare Two Specific Architectures
```python
from experiments import ExperimentRunner
from loader import DataLoader
from error import cross_entropy
import activation as act

loader = DataLoader("./MNIST")
X_train, Y_train, X_test, Y_test = loader.get_train_test_data()
runner = ExperimentRunner(loader, X_test, Y_test)

# Test shallow network
result_shallow = runner.run_single_experiment(
    input_neurons=784, hidden_neurons=[256], output_neurons=10,
    eta=0.1, epoch_number=200, patience=10,
    error_function=cross_entropy,
    activation_function=act.relu, activation_name="relu",
    use_kfold=False
)

# Test deep network
result_deep = runner.run_single_experiment(
    input_neurons=784, hidden_neurons=[128, 64, 32], output_neurons=10,
    eta=0.1, epoch_number=200, patience=10,
    error_function=cross_entropy,
    activation_function=act.relu, activation_name="relu",
    use_kfold=False
)

# Compare
print(f"Shallow [256]: {result_shallow['test_accuracy']:.2%}")
print(f"Deep [128,64,32]: {result_deep['test_accuracy']:.2%}")
print("✓ Both results saved to experiments_results.json")
```

### Example 3: Manual Comparison with Custom Saving
```python
from network import Network
from loader import DataLoader
from trainer import Trainer
from error import cross_entropy
from results_manager import ResultsManager
import activation as act

loader = DataLoader("./MNIST")
X_train, Y_train, X_test, Y_test = loader.get_train_test_data()

# Collect results
results = []

# Test different architectures
for hidden in [[50], [100], [128, 64]]:
    network = Network(784, hidden, 10, activation_function=act.relu)
    history = Trainer.holdout_validation(
        loader, network, cross_entropy,
        epoch_number=100, eta=0.1, patience=10
    )
    
    Z_test = network.forward_propagation(X_test)
    test_acc = network.get_accuracy(Z_test, Y_test)
    
    result = {
        "hidden_neurons": hidden,
        "eta": 0.1,
        "activation": "relu",
        "epoch_number": 100,
        "patience": 10,
        "method": "holdout",
        "test_accuracy": float(test_acc),
        "epochs_trained": history["epochs_trained"],
        "final_valid_accuracy": history["final_valid_accuracy"],
        "status": "success"
    }
    results.append(result)
    print(f"Architecture {hidden}: {test_acc:.2%}")

# Save all results at once
ResultsManager.save_json(results, "architecture_comparison.json")
ResultsManager.save_csv(results, "architecture_comparison.csv")
ResultsManager.generate_markdown_report(results, "ARCHITECTURE_COMPARISON.md")
print("\n✓ Results saved in JSON, CSV, and Markdown formats")
```

### Example 4: Experiment with K-Fold Validation
```python
from experiments import ExperimentRunner
from loader import DataLoader
from error import cross_entropy
import activation as act

loader = DataLoader("./MNIST")
X_train, Y_train, X_test, Y_test = loader.get_train_test_data()
runner = ExperimentRunner(loader, X_test, Y_test)

# Run with K-fold cross-validation (automatically saved)
result = runner.run_single_experiment(
    input_neurons=784,
    hidden_neurons=[128],
    output_neurons=10,
    eta=0.1,
    epoch_number=200,
    patience=10,
    error_function=cross_entropy,
    activation_function=act.relu,
    activation_name="relu",
    use_kfold=True,               # Enable K-fold
    k=5                            # 5 folds
)

# K-fold provides averaged metrics
print(f"Test Accuracy: {result['test_accuracy']:.2%}")
print(f"Avg Validation Accuracy (5 folds): {result['final_valid_accuracy']:.2%}")
print("✓ Results automatically saved to experiments_results.json")
```

### Example 5: Visualize During Training
```python
from network import Network
from loader import DataLoader
from trainer import Trainer
from error import cross_entropy
from visualization import Visualizer
from results_manager import ResultsManager
import activation as act

loader = DataLoader("./MNIST")
X_train, Y_train, X_test, Y_test = loader.get_train_test_data()

# Train network
network = Network(784, [128], 10, activation_function=act.relu)
history = Trainer.holdout_validation(
    loader, network, cross_entropy,
    epoch_number=100, eta=0.1, patience=10
)

# Test and save
Z_test = network.forward_propagation(X_test)
test_accuracy = network.get_accuracy(Z_test, Y_test)
print(f"Test Accuracy: {test_accuracy:.2%}")

# Save result
result = {
    "hidden_neurons": [128],
    "eta": 0.1,
    "activation": "relu",
    "test_accuracy": float(test_accuracy),
    "final_valid_accuracy": history["final_valid_accuracy"],
    "epochs_trained": history["epochs_trained"],
    "status": "success"
}
ResultsManager.append_result(result, "training_with_viz.json")

# Show predictions
print("\nVisualizing predictions...")
Visualizer.show_multiple_predictions(network, X_test, Y_test, num_samples=10)
```

---

## Project Structure
```
neural-network-MNIST-2025/
├── main.py                 # Main experiment pipeline (auto-saves results)
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

---

## Output Files

Generated after running experiments:

- **`experiments_results.json`**: Structured results with full details (default format)
- **`experiments_results.csv`**: Tabular format for analysis (open in Excel)
- **`RESULTS.md`**: Human-readable markdown report

**Result Saving Summary:**

| Method | Auto-Save? | How to Save Manually |
|--------|------------|---------------------|
| `main.py` | Yes | N/A |
| `ExperimentRunner.run_single_experiment()` | Yes | N/A |
| `Trainer.holdout_validation()` | No | Use `ResultsManager.append_result()` |
| `Trainer.kfold_cross_validation()` | No | Use `ResultsManager.append_result()` |
| `Network.fit()` | No | Use `ResultsManager.append_result()` |

---

## Requirements

- **Python**: 3.7+
- **NumPy**: Numerical computations
- **Matplotlib**: Visualization

See `requirements.txt` for specific versions.

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.
