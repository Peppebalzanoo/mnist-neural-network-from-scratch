"""
Examples of how to use the neural network implementation.

This file contains various examples demonstrating different ways to train
and evaluate neural networks on the MNIST dataset.
"""

from network import Network
from loader import DataLoader
from trainer import Trainer
from error import cross_entropy
from experiments import ExperimentRunner
from visualization import Visualizer
import activation as act


# ============================================================================
# EXAMPLE 1: Direct Training with network.fit()
# ============================================================================
def example_direct_training():
    """
    Train a network directly using network.fit().
    You have full control over data splitting.
    """
    print("\n" + "="*80)
    print("EXAMPLE 1: Direct Training with network.fit()")
    print("="*80 + "\n")
    
    # Load data
    loader = DataLoader("./MNIST")
    X_train, Y_train, X_test, Y_test = loader.get_train_test_data()

    # Split data into train and validation sets
    train_ratio = 0.8
    X_train_split, Y_train_split, X_valid, Y_valid = loader.split_data(train_ratio)

    # Create network: 784 inputs -> 128 hidden -> 10 outputs
    network = Network(784, [128], 10, activation_function=act.relu)

    # Train the network directly
    history = network.fit(
        X_train=X_train_split,
        Y_train=Y_train_split,
        X_valid=X_valid,
        Y_valid=Y_valid,
        error_function=cross_entropy,
        epoch_number=100,
        eta=0.01,
        patience=10
    )

    # Evaluate on test set
    Z_test = network.forward_propagation(X_test)
    test_accuracy = network.get_accuracy(Z_test, Y_test)
    print(f"\nTest Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"Training stopped at epoch: {history['epochs_trained']}")
    print(f"Final Validation Accuracy: {history['final_valid_accuracy']:.4f}")


# ============================================================================
# EXAMPLE 2: Using Trainer.holdout_validation()
# ============================================================================
def example_holdout_validation():
    """
    Train using Trainer.holdout_validation().
    The Trainer handles data splitting automatically (faster).
    """
    print("\n" + "="*80)
    print("EXAMPLE 2: Using Trainer.holdout_validation()")
    print("="*80 + "\n")
    
    # Load data
    loader = DataLoader("./MNIST")
    X_train, Y_train, X_test, Y_test = loader.get_train_test_data()

    # Create network
    network = Network(784, [128], 10, activation_function=act.relu)

    # Train using Trainer (handles data splitting automatically)
    history = Trainer.holdout_validation(
        loader=loader,
        network=network,
        error_function=cross_entropy,
        epoch_number=100,
        eta=0.01,
        patience=10,
        train_ratio=0.8  # Trainer splits the data for you
    )

    # Evaluate on test set
    Z_test = network.forward_propagation(X_test)
    test_accuracy = network.get_accuracy(Z_test, Y_test)
    print(f"\nTest Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"Final Validation Accuracy: {history['final_valid_accuracy']:.4f}")


# ============================================================================
# EXAMPLE 3: Using Trainer.kfold_cross_validation()
# ============================================================================
def example_kfold_cross_validation():
    """
    Train using K-Fold Cross-Validation.
    Most robust evaluation (slower but more reliable).
    """
    print("\n" + "="*80)
    print("EXAMPLE 3: Using Trainer.kfold_cross_validation()")
    print("="*80 + "\n")
    
    # Load data
    loader = DataLoader("./MNIST")
    X_train, Y_train, X_test, Y_test = loader.get_train_test_data()

    # Train using K-Fold Cross-Validation
    # Returns the best model among all folds
    best_network, avg_history = Trainer.kfold_cross_validation(
        loader=loader,
        input_neurons=784,
        hidden_neurons=[128],
        output_neurons=10,
        error_function=cross_entropy,
        k=5,  # 5-fold cross-validation
        epoch_number=100,
        eta=0.01,
        patience=10,
        activation_function=act.relu
    )

    # Evaluate on test set
    Z_test = best_network.forward_propagation(X_test)
    test_accuracy = best_network.get_accuracy(Z_test, Y_test)
    print(f"\nTest Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"Average Final Validation Accuracy: {avg_history['final_valid_accuracy']:.4f}")


# ============================================================================
# EXAMPLE 4: Run a Custom Experiment
# ============================================================================
def example_custom_experiment():
    """
    Run a single experiment with specific configuration using ExperimentRunner.
    
    DIFFERENCE from Examples 1, 2, 3:
    - Examples 1-3: Train a network and get results in memory only
    - Example 4: Run a full experiment that SAVES results to JSON file
    
    ExperimentRunner adds:
    - Automatic result saving to experiments_results.json
    - Complete experiment metadata (config, metrics, status)
    - Structured format for comparing multiple experiments
    - Used internally by main.py for grid search experiments
    
    Use this when you want to TRACK and COMPARE experiments,
    not just train a single network for immediate use.
    """
    print("\n" + "="*80)
    print("EXAMPLE 4: Run a Custom Experiment")
    print("="*80 + "\n")
    
    # Setup
    loader = DataLoader("./MNIST")
    X_train, Y_train, X_test, Y_test = loader.get_train_test_data()
    runner = ExperimentRunner(loader, X_test, Y_test)

    # Run single experiment with specific configuration
    result = runner.run_single_experiment(
        input_neurons=784,
        hidden_neurons=[100, 50],
        output_neurons=10,
        eta=1.0,
        epoch_number=500,
        patience=10,
        error_function=cross_entropy,
        activation_function=act.relu,
        activation_name="relu",
        use_kfold=False
    )

    print(f"\nTest Accuracy: {result['test_accuracy']:.4f}")
    print(f"Final Validation Accuracy: {result['final_valid_accuracy']:.4f}")
    print(f"Epochs Trained: {result['epochs_trained']}")
    
    # Save results to JSON file
    from results_manager import ResultsManager
    results_manager = ResultsManager()
    results_manager.save_json([result], "experiments_custom_results.json")
    print(f"\n✓ Results saved to experiments_custom_results.json")


# ============================================================================
# EXAMPLE 5: Visualize Predictions
# ============================================================================
def example_visualize_predictions():
    """
    Train a network and visualize its predictions on test samples.
    """
    print("\n" + "="*80)
    print("EXAMPLE 5: Visualize Predictions")
    print("="*80 + "\n")
    
    # Load data
    loader = DataLoader("./MNIST")
    X_train, Y_train, X_test, Y_test = loader.get_train_test_data()

    # Create and train network
    network = Network(784, [128], 10, activation_function=act.relu)
    
    print("Training network...")
    history = Trainer.holdout_validation(
        loader=loader,
        network=network,
        error_function=cross_entropy,
        epoch_number=50,
        eta=1.0,
        patience=10,
        train_ratio=0.8
    )

    # Show predictions on random test samples
    print("\nVisualizing predictions...")
    Visualizer.show_multiple_predictions(network, X_test, Y_test, num_samples=5)

    # Show a single prediction with specific seed
    Visualizer.show_prediction(network, X_test, Y_test, random_seed=42)


# ============================================================================
# EXAMPLE 6: Comparing Different Activation Functions
# ============================================================================
def example_compare_activations():
    """
    Compare different activation functions on the same architecture.
    """
    print("\n" + "="*80)
    print("EXAMPLE 6: Comparing Different Activation Functions")
    print("="*80 + "\n")
    
    loader = DataLoader("./MNIST")
    X_train, Y_train, X_test, Y_test = loader.get_train_test_data()
    
    activations = [
        (act.sigmoid, "sigmoid"),
        (act.tanh, "tanh"),
        (act.relu, "relu")
    ]
    
    results = []
    
    for activation_func, activation_name in activations:
        print(f"\nTesting {activation_name} activation...")
        
        network = Network(784, [100], 10, activation_function=activation_func)
        
        history = Trainer.holdout_validation(
            loader=loader,
            network=network,
            error_function=cross_entropy,
            epoch_number=50,
            eta=0.01,
            patience=10,
            train_ratio=0.8
        )
        
        Z_test = network.forward_propagation(X_test)
        test_accuracy = network.get_accuracy(Z_test, Y_test)
        
        results.append({
            "activation": activation_name,
            "test_accuracy": test_accuracy,
            "final_valid_accuracy": history['final_valid_accuracy']
        })
    
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)
    for r in results:
        print(f"{r['activation']:10} - Test: {r['test_accuracy']:.4f}, Valid: {r['final_valid_accuracy']:.4f}")


# ============================================================================
# EXAMPLE 7: Comparing Different Architectures
# ============================================================================
def example_compare_architectures():
    """
    Compare different network architectures.
    """
    print("\n" + "="*80)
    print("EXAMPLE 7: Comparing Different Architectures")
    print("="*80 + "\n")
    
    loader = DataLoader("./MNIST")
    X_train, Y_train, X_test, Y_test = loader.get_train_test_data()
    
    architectures = [
        [50],
        [100],
        [100, 50],
        [128, 64]
    ]
    
    results = []
    
    for hidden_neurons in architectures:
        print(f"\nTesting architecture: 784 -> {hidden_neurons} -> 10")
        
        network = Network(784, hidden_neurons, 10, activation_function=act.relu)
        
        history = Trainer.holdout_validation(
            loader=loader,
            network=network,
            error_function=cross_entropy,
            epoch_number=50,
            eta=0.01,
            patience=10,
            train_ratio=0.8
        )
        
        Z_test = network.forward_propagation(X_test)
        test_accuracy = network.get_accuracy(Z_test, Y_test)
        
        results.append({
            "architecture": str(hidden_neurons),
            "test_accuracy": test_accuracy,
            "epochs_trained": history['epochs_trained']
        })
    
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)
    for r in results:
        print(f"{r['architecture']:20} - Test: {r['test_accuracy']:.4f}, Epochs: {r['epochs_trained']}")
# ============================================================================
# END OF EXAMPLES   

if __name__ == "__main__":
    # example_direct_training()
    # example_holdout_validation()
    # example_kfold_cross_validation()
    example_custom_experiment()
    # example_visualize_predictions()
    # example_compare_activations()
    # example_compare_architectures()