import numpy as np
from loader import DataLoader
from network import Network
from error import cross_entropy
from config import ExperimentConfig
from experiments import ExperimentRunner
from results_manager import ResultsManager
from visualization import Visualizer
import activation as act


def simple_network_example():
    """
    EXAMPLE: Train and evaluate a single network without running full experiments.
    
    This function demonstrates how to:
    1. Load the MNIST dataset
    2. Create a custom network architecture
    3. Train the network with specific hyperparameters
    4. Evaluate performance on test set
    5. Visualize predictions (optional)
    
    Uncomment the function call at the bottom of this file to run this example.
    """
    print("=" * 80)
    print("SIMPLE NETWORK TRAINING EXAMPLE")
    print("=" * 80)
    
    # Load dataset
    print("\nLoading MNIST dataset...")
    loader = DataLoader("./MNIST")
    X_train, Y_train, X_test, Y_test = loader.get_train_test_data()
    print(f"Dataset loaded: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
    
    # Define network architecture
    input_neurons = X_train.shape[1]   # 784 (28x28 pixels)
    hidden_neurons = [128, 64]         # Two hidden layers: 128 -> 64
    output_neurons = Y_train.shape[1]  # 10 (digits 0-9)
    
    print(f"\nNetwork Architecture: {input_neurons} -> {hidden_neurons} -> {output_neurons}")
    
    # Create network with ReLU activation
    network = Network(
        input_neurons=input_neurons,
        hiddens_neurons=hidden_neurons,
        output_neurons=output_neurons,
        activation_function=act.relu  # Try: act.sigmoid, act.tanh, act.relu
    )
    
    # Split data into train and validation sets
    train_ratio = 0.8
    X_train_split, Y_train_split, X_valid, Y_valid = loader.split_data(train_ratio)
    print(f"Split: {X_train_split.shape[0]} training, {X_valid.shape[0]} validation")
    
    # Print network information
    network.get_info()
    
    # Train the network
    print("\nTraining network...")
    print("Hyperparameters: eta=0.01, max_epochs=100, patience=5")
    
    network.fit(
        X_train=X_train_split,
        Y_train=Y_train_split,
        X_valid=X_valid,
        Y_valid=Y_valid,
        error_function=cross_entropy,
        epoch_number=100,       # Maximum epochs
        eta=0.01,               # Learning rate
        patience=5              # Early stopping patience
    )
    
    # Evaluate on test set
    print("\n" + "-" * 80)
    print("EVALUATION RESULTS")
    print("-" * 80)
    
    Z_test = network.forward_propagation(X_test)
    test_accuracy = network.get_accuracy(Z_test, Y_test)
    test_loss = cross_entropy(Z_test, Y_test)
    
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"Test Loss: {test_loss:.6f}")
    
    # Optional: Visualize some predictions
    # Uncomment the lines below to see visual predictions
    # print("\nVisualizing predictions...")
    # Visualizer.show_multiple_predictions(network, X_test, Y_test, num_samples=5)
    
    print("\n" + "=" * 80)
    print("Example completed! Modify this function to experiment with:")
    print("  - Different architectures: hidden_neurons = [50], [100, 50], [128, 64, 32]")
    print("  - Different learning rates: eta = 0.001, 0.01, 0.1")
    print("  - Different activations: act.sigmoid, act.tanh, act.relu")
    print("=" * 80 + "\n")
    
    return network


def main():
    # ========================================================================
    # DATASET LOADING
    # ========================================================================
    # Load MNIST dataset from the specified directory.
    # Change "./MNIST" path if your dataset is located elsewhere.
    print("Loading MNIST dataset...")
    loader = DataLoader("./MNIST")
    X_train, Y_train, X_test, Y_test = loader.get_train_test_data()
    print(f"Dataset loaded: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples\n")

    # ========================================================================
    # EXPERIMENT CONFIGURATION
    # ========================================================================
    # Network size is determined by the dataset:
    # - Input neurons = number of features (784 for MNIST 28x28 images)
    # - Output neurons = number of classes (10 for digits 0-9)
    input_neurons = X_train.shape[1]
    output_neurons = Y_train.shape[1]
    
    # Load experiment parameters from config.py
    # MODIFY config.py TO CHANGE: architectures, learning rates, epochs, etc.
    config = ExperimentConfig()

    # ========================================================================
    # EXPERIMENT EXECUTION
    # ========================================================================
    # Create experiment runner with the loaded dataset
    runner = ExperimentRunner(loader, X_test, Y_test)

    # Step 1: Run holdout validation experiments
    # Tests ALL combinations of: architectures × learning_rates × activation_functions
    # This initial screening identifies the most promising configurations.
    # Results are saved incrementally after each experiment (safe for interruptions).
    holdout_results = runner.run_holdout_experiments(
        input_neurons=input_neurons,
        output_neurons=output_neurons,
        hidden_configurations=config.HIDDEN_CONFIGURATIONS,
        learning_rates=config.LEARNING_RATES,
        activation_functions=config.ACTIVATION_FUNCTIONS,
        error_function=cross_entropy,
        epoch_number=config.EPOCH_NUMBER,
        patience=config.PATIENCE,
    )

    # Step 2: Run k-fold cross-validation on top configurations
    # Re-evaluates the best N configurations from holdout using k-fold CV
    # for more robust performance estimates. Adjust TOP_N_FOR_KFOLD in config.py.
    # Results are saved incrementally after each k-fold experiment.
    kfold_results = runner.run_kfold_on_best(
        input_neurons=input_neurons,
        output_neurons=output_neurons,
        holdout_results=holdout_results,
        activation_functions=config.ACTIVATION_FUNCTIONS,
        error_function=cross_entropy,
        epoch_number=config.EPOCH_NUMBER,
        patience=config.PATIENCE,
        k=config.K_FOLDS,
        top_n=config.TOP_N_FOR_KFOLD,
    )

    # ========================================================================
    # RESULTS PROCESSING AND SAVING
    # ========================================================================
    # Combine holdout and k-fold results for comprehensive analysis
    all_results = holdout_results + kfold_results

    # Save results in multiple formats (JSON, CSV, Markdown)
    # Output file names can be changed in config.py
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)

    ResultsManager.save_json(all_results, config.JSON_OUTPUT)
    ResultsManager.save_csv(all_results, config.CSV_OUTPUT)
    ResultsManager.generate_markdown_report(all_results, config.MARKDOWN_OUTPUT)

    # Print summary statistics to console
    ResultsManager.print_summary(all_results)

    # ========================================================================
    # OPTIONAL: VISUALIZATION OF BEST MODEL
    # ========================================================================
    # Uncomment the code below to train the best model and visualize its predictions
    # on test samples. Useful for qualitative analysis of model performance.
    # successful = [r for r in all_results if r["status"] == "success"]
    # if successful:
    #     best = max(successful, key=lambda x: x["test_accuracy"])
    #     print("\nTraining best model for visualization...")
    #     activation_func = next(f for f, n in config.ACTIVATION_FUNCTIONS if n == best["activation"])
    #     best_model = Network(input_neurons, best["hidden_neurons"], output_neurons, activation_function=activation_func)
    #     # Train and visualize...
    #     Visualizer.show_multiple_predictions(best_model, X_test, Y_test, num_samples=10)


if __name__ == "__main__":
    # ========================================================================
    # CHOOSE ONE OF THE FOLLOWING OPTIONS:
    # ========================================================================
    
    # OPTION 1: Run the simple network example (single network training)
    # Uncomment the line below to train a single network without grid search
    # simple_network_example()
    
    # OPTION 2: Run full experimental pipeline (default)
    # Grid search over architectures, learning rates, and activation functions
    main()