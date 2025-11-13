import activation as act


class ExperimentConfig:
    """Configuration for neural network experiments.
    
    MODIFY THIS FILE TO CHANGE SIMULATION PARAMETERS:
    - Update network architectures, learning rates, and activation functions
    - Adjust training settings (epochs, early stopping patience)
    - Configure validation strategy (k-fold, train/test split ratio)
    - Change output file names and formats
    """

    # ========================================================================
    # NETWORK ARCHITECTURE CONFIGURATIONS
    # ========================================================================
    # Define different hidden layer architectures to test.
    # Each list represents one configuration: [neurons_layer1, neurons_layer2, ...]
    # Add or remove configurations to test different network depths and widths.
    # Example: [200] for single layer, [100, 50] for two layers, etc.
    HIDDEN_CONFIGURATIONS = [
        [50],  # Single hidden layer with 50 neurons
        [100],  # Single hidden layer with 100 neurons
        # [128],  # Single hidden layer with 128 neurons
        # [50, 50],  # Two hidden layers with 50 neurons each
        # [100, 50],  # Two hidden layers: 100 -> 50
        # [128, 64],  # Two hidden layers: 128 -> 64
        # [100, 50, 25],  # Three hidden layers: 100 -> 50 -> 25
        # [128, 64, 32],  # Three hidden layers: 128 -> 64 -> 32
    ]

    # ========================================================================
    # HYPERPARAMETERS TO TEST
    # ========================================================================
    # Learning rates control how much the network adjusts weights during training.
    # Smaller values (0.001-0.01) are safer but slower; larger values (0.1-0.5) are faster but riskier.
    # Testing with higher learning rates to see if networks learn better.
    LEARNING_RATES = [0.1, 0.5, 1.0]

    # Activation functions to apply in hidden layers.
    # Each tuple is: (function_object, "function_name")
    # Add/remove functions from activation.py to test different non-linearities.
    ACTIVATION_FUNCTIONS = [
        (act.sigmoid, "sigmoid"),
        (act.tanh, "tanh"),
        (act.relu, "relu"),
    ]

    # ========================================================================
    # TRAINING PARAMETERS
    # ========================================================================
    # Maximum number of training epochs (iterations over the entire dataset).
    # Increase for more thorough training, decrease for faster experiments.
    EPOCH_NUMBER = 1000
    
    # Early stopping patience: training stops if validation loss doesn't improve
    # for this many consecutive epochs. Prevents overfitting and saves time.
    PATIENCE = 10
    
    # Train/test split ratio for holdout validation (0.8 = 80% train, 20% test).
    # Only used in initial holdout experiments, not in k-fold validation.
    TRAIN_RATIO = 0.8
    
    # Number of folds for k-fold cross-validation on the best configurations.
    # More folds = more reliable results but longer computation time.
    K_FOLDS = 5
    
    # How many top configurations from holdout to re-evaluate with k-fold.
    # The best N configurations will undergo more rigorous k-fold testing.
    TOP_N_FOR_KFOLD = 3

    # ========================================================================
    # OUTPUT CONFIGURATION
    # ========================================================================
    # File names for saving experiment results.
    # Change these to organize results from different experiment runs.
    JSON_OUTPUT = "experiments_results.json"  # Structured data with all details
    CSV_OUTPUT = "experiments_results.csv"     # Tabular format for spreadsheet analysis
    MARKDOWN_OUTPUT = "RESULTS.md"              # Human-readable report