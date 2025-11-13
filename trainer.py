import numpy as np
from network import Network
from loader import DataLoader


class Trainer:
    """Handle different training strategies for neural networks."""

    @staticmethod
    def holdout_validation(
        loader: DataLoader,
        network: Network,
        error_function,
        epoch_number: int,
        eta: float,
        patience: int,
        train_ratio: float = 0.8,
    ) -> dict:
        """Train the network using holdout validation and return training history."""
        X_train, Y_train, X_valid, Y_valid = loader.split_data(train_ratio)
        network.get_info()
        history = network.fit(X_train, Y_train, X_valid, Y_valid, error_function, epoch_number, eta, patience)
        Z_valid = network.forward_propagation(X_valid)
        print(f"Validation Set accuracy: {network.get_accuracy(Z_valid, Y_valid):.15f}")
        return history

    @staticmethod
    def kfold_cross_validation(
        loader,
        input_neurons: int,
        hidden_neurons: list,
        output_neurons: int,
        error_function,
        k: int,
        epoch_number: int,
        eta: float,
        patience: int,
        activation_function=None,
    ) -> tuple:
        """Train using k-fold cross-validation and return the best model and aggregated history."""
        validation_accuracies = []
        best_model = None
        best_accuracy = 0.0
        fold_histories = []

        fold_indices = loader.create_stratified_k_folds(loader._train_labels, k)

        for i in range(k):
            print(f"\n--- Start Fold {i+1}/{k} ---")

            val_indices = fold_indices[i]
            train_indices_list = [fold_indices[j] for j in range(k) if j != i]
            train_indices = np.concatenate(train_indices_list)

            if loader._train_data is None or loader._train_labels is None:
                raise ValueError("Training data or labels not loaded. Please call _load_data() first.")

            X_train, Y_train = loader._train_data[train_indices], loader._train_labels[train_indices]
            X_valid, Y_valid = loader._train_data[val_indices], loader._train_labels[val_indices]

            print(f"Training Set Size: {X_train.shape[0]}")
            print(f"Validation Set Size: {X_valid.shape[0]}")

            network = Network(input_neurons, hidden_neurons, output_neurons, activation_function=activation_function)

            history = network.fit(X_train, Y_train, X_valid, Y_valid, error_function, epoch_number, eta, patience)
            fold_histories.append(history)

            Z_valid = network.forward_propagation(X_valid)
            curr_acc = network.get_accuracy(Z_valid, Y_valid)
            validation_accuracies.append(curr_acc)

            if curr_acc > best_accuracy:
                best_accuracy = curr_acc
                best_model = network.copy_network()

            print(f"Validation Accuracy of Fold {i+1}: {curr_acc:.15f}")

        mean_accuracy = np.mean(validation_accuracies)
        std_accuracy = np.std(validation_accuracies)
        print(f"\nK-Fold Cross-Validation Results: Mean Accuracy = {mean_accuracy:.15f}, Std Dev = {std_accuracy:.15f}")

        if best_model is None:
            raise RuntimeError("No valid model was trained during k-fold cross-validation.")
        
        # Calculate average metrics across folds
        avg_history = {
            "epochs_trained": np.mean([h["epochs_trained"] for h in fold_histories]),
            "initial_valid_error": np.mean([h["initial_valid_error"] for h in fold_histories]),
            "initial_valid_accuracy": np.mean([h["initial_valid_accuracy"] for h in fold_histories]),
            "final_valid_error": np.mean([h["final_valid_error"] for h in fold_histories]),
            "final_valid_accuracy": np.mean([h["final_valid_accuracy"] for h in fold_histories]),
            "best_valid_error": np.mean([h["best_valid_error"] for h in fold_histories])
        }

        return best_model, avg_history