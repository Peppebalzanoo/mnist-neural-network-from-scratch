import numpy as np
from network import Network
from loader import DataLoader
from trainer import Trainer
from results_manager import ResultsManager


class ExperimentRunner:
    """Handle running and managing experiments."""

    def __init__(self, loader: DataLoader, X_test: np.ndarray, Y_test: np.ndarray):
        self.loader = loader
        self.X_test = X_test
        self.Y_test = Y_test

    def run_single_experiment(
        self,
        input_neurons: int,
        hidden_neurons: list,
        output_neurons: int,
        eta: float,
        epoch_number: int,
        patience: int,
        error_function,
        activation_function=None,
        activation_name: str = "sigmoid",
        use_kfold: bool = False,
        k: int = 5,
    ) -> dict:
        """Run a single experiment and return results."""
        method = "k-fold" if use_kfold else "holdout"

        print(f"\n{'='*80}")
        print(f"EXPERIMENT: Hidden={hidden_neurons}, eta={eta}, Activation={activation_name}")
        print(f"Method: {method.upper()}")
        print(f"{'='*80}")

        experiment_result = {
            "hidden_neurons": hidden_neurons,
            "eta": eta,
            "activation": activation_name,
            "epoch_number": epoch_number,
            "patience": patience,
            "method": method,
        }

        try:
            if use_kfold:
                best_model, history = Trainer.kfold_cross_validation(
                    self.loader,
                    input_neurons,
                    hidden_neurons,
                    output_neurons,
                    error_function,
                    k,
                    epoch_number,
                    eta,
                    patience,
                    activation_function,
                )
                experiment_result["k_folds"] = k
            else:
                mynet = Network(input_neurons, hidden_neurons, output_neurons, activation_function=activation_function)
                history = Trainer.holdout_validation(
                    self.loader, mynet, error_function, epoch_number, eta, patience, train_ratio=0.8
                )
                best_model = mynet

            # Test on test set
            Z_test = best_model.forward_propagation(self.X_test)
            test_accuracy = best_model.get_accuracy(Z_test, self.Y_test)

            # Save test accuracy
            experiment_result["test_accuracy"] = float(test_accuracy)
            
            # Save validation metrics from training history
            experiment_result["epochs_trained"] = int(history["epochs_trained"])
            experiment_result["initial_valid_error"] = float(history["initial_valid_error"])
            experiment_result["initial_valid_accuracy"] = float(history["initial_valid_accuracy"])
            experiment_result["final_valid_error"] = float(history["final_valid_error"])
            experiment_result["final_valid_accuracy"] = float(history["final_valid_accuracy"])
            experiment_result["best_valid_error"] = float(history["best_valid_error"])
            
            experiment_result["status"] = "success"

            print(f"\n{'='*80}")
            print(f"TEST SET ACCURACY: {test_accuracy:.6f} ({test_accuracy*100:.2f}%)")
            print(f"{'='*80}\n")

        except Exception as e:
            print(f"\nERROR in experiment: {str(e)}\n")
            experiment_result["test_accuracy"] = 0.0
            experiment_result["status"] = "failed"
            experiment_result["error"] = str(e)

        return experiment_result

    def run_holdout_experiments(
        self,
        input_neurons: int,
        output_neurons: int,
        hidden_configurations: list,
        learning_rates: list,
        activation_functions: list,
        error_function,
        epoch_number: int,
        patience: int,
        save_incremental: bool = True,
        results_file: str = "experiments_results.json",
    ) -> list:
        """Run all holdout validation experiments."""
        results = []
        total_experiments = len(hidden_configurations) * len(learning_rates) * len(activation_functions)

        print("\n" + "=" * 80)
        print("RUNNING HOLDOUT VALIDATION EXPERIMENTS")
        print(f"Total experiments: {total_experiments}")
        print("=" * 80)

        experiment_count = 0

        for hidden_neurons in hidden_configurations:
            for eta in learning_rates:
                for activation_func, activation_name in activation_functions:
                    experiment_count += 1
                    print(f"\n[Experiment {experiment_count}/{total_experiments}]")

                    result = self.run_single_experiment(
                        input_neurons=input_neurons,
                        hidden_neurons=hidden_neurons,
                        output_neurons=output_neurons,
                        eta=eta,
                        epoch_number=epoch_number,
                        patience=patience,
                        error_function=error_function,
                        activation_function=activation_func,
                        activation_name=activation_name,
                        use_kfold=False,
                    )
                    
                    results.append(result)
                    
                    # Save incrementally after each experiment
                    if save_incremental:
                        results_manager = ResultsManager()
                        results_manager.append_result(result, results_file)

        return results

    def run_kfold_on_best(
        self,
        input_neurons: int,
        output_neurons: int,
        holdout_results: list,
        activation_functions: list,
        error_function,
        epoch_number: int,
        patience: int,
        k: int = 5,
        top_n: int = 3,
        save_incremental: bool = True,
        results_file: str = "experiments_results.json",
    ) -> list:
        """Run k-fold cross-validation on the best configurations from holdout."""
        results = []
        successful_results = [r for r in holdout_results if r["status"] == "success"]

        if not successful_results:
            print("\nWarning: No successful holdout experiments to select from.")
            return results

        print("\n" + "=" * 80)
        print(f"RUNNING K-FOLD CROSS-VALIDATION EXPERIMENTS (Top {top_n} Configurations)")
        print("=" * 80)

        # Get top N configurations
        successful_results.sort(key=lambda x: x["test_accuracy"], reverse=True)
        top_configs = [(r["hidden_neurons"], r["eta"], r["activation"]) for r in successful_results[:top_n]]

        for i, (hidden_neurons, eta, activation_name) in enumerate(top_configs, 1):
            print(f"\n[K-Fold Experiment {i}/{top_n}]")

            # Find the activation function object
            activation_func = next(func for func, name in activation_functions if name == activation_name)

            result = self.run_single_experiment(
                input_neurons=input_neurons,
                hidden_neurons=hidden_neurons,
                output_neurons=output_neurons,
                eta=eta,
                epoch_number=epoch_number,
                patience=patience,
                error_function=error_function,
                activation_function=activation_func,
                activation_name=activation_name,
                use_kfold=True,
                k=k,
            )
            
            if result is not None:
                results.append(result)
                
                # Save incrementally after each k-fold experiment
                if save_incremental:
                    results_manager = ResultsManager()
                    results_manager.append_result(result, results_file)

        return results