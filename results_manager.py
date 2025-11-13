import json
import csv
import numpy as np
from datetime import datetime


class ResultsManager:
    """Handle saving and reporting experiment results."""

    @staticmethod
    def save_json(results: list, filename: str = "experiments_results.json"):
        """Save experiment results to a JSON file."""
        output = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_experiments": len(results),
            "experiments": results,
        }

        with open(filename, "w") as f:
            json.dump(output, f, indent=2)

        print(f"\nResults saved to {filename}")

    @staticmethod
    def append_result(result: dict, filename: str = "experiments_results.json"):
        """Append a single experiment result to the JSON file incrementally."""
        # Try to load existing results
        try:
            with open(filename, "r") as f:
                data = json.load(f)
                existing_results = data.get("experiments", [])
        except (FileNotFoundError, json.JSONDecodeError):
            # File doesn't exist or is empty, start fresh
            existing_results = []
        
        # Append new result
        existing_results.append(result)
        
        # Save updated results
        output = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_experiments": len(existing_results),
            "experiments": existing_results,
        }
        
        with open(filename, "w") as f:
            json.dump(output, f, indent=2)
        
        print(f"Result saved to {filename} (total: {len(existing_results)} experiments)")

    @staticmethod
    def save_csv(results: list, filename: str = "experiments_results.csv"):
        """Save experiment results to a CSV file."""
        if not results:
            return

        fieldnames = [
            "hidden_neurons",
            "eta",
            "activation",
            "method",
            "epoch_number",
            "patience",
            "epochs_trained",
            "initial_valid_error",
            "initial_valid_accuracy",
            "final_valid_error",
            "final_valid_accuracy",
            "best_valid_error",
            "test_accuracy",
            "test_accuracy_percent",
            "status",
        ]

        with open(filename, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for result in results:
                row = {
                    "hidden_neurons": str(result["hidden_neurons"]),
                    "eta": result["eta"],
                    "activation": result["activation"],
                    "method": result["method"],
                    "epoch_number": result["epoch_number"],
                    "patience": result["patience"],
                    "epochs_trained": result.get("epochs_trained", "N/A"),
                    "initial_valid_error": f"{result.get('initial_valid_error', 0):.6f}" if result.get("initial_valid_error") is not None else "N/A",
                    "initial_valid_accuracy": f"{result.get('initial_valid_accuracy', 0):.6f}" if result.get("initial_valid_accuracy") is not None else "N/A",
                    "final_valid_error": f"{result.get('final_valid_error', 0):.6f}" if result.get("final_valid_error") is not None else "N/A",
                    "final_valid_accuracy": f"{result.get('final_valid_accuracy', 0):.6f}" if result.get("final_valid_accuracy") is not None else "N/A",
                    "best_valid_error": f"{result.get('best_valid_error', 0):.6f}" if result.get("best_valid_error") is not None else "N/A",
                    "test_accuracy": f"{result['test_accuracy']:.6f}",
                    "test_accuracy_percent": f"{result['test_accuracy']*100:.2f}%",
                    "status": result["status"],
                }
                writer.writerow(row)

        print(f"Results saved to {filename}")

    @staticmethod
    def generate_markdown_report(results: list, filename: str = "RESULTS.md"):
        """Generate a detailed markdown report from results."""
        markdown = "# Neural Network Experiments Report\n\n"
        markdown += f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        markdown += f"**Total Experiments:** {len(results)}\n\n"

        successful_results = [r for r in results if r["status"] == "success"]

        if not successful_results:
            markdown += "Warning: No successful experiments to report.\n"
            with open(filename, "w") as f:
                f.write(markdown)
            return

        # Experiment Results Table
        markdown += "---\n\n## Experiment Results\n\n"
        markdown += "| Hidden Layers | Activation | Eta | Method | Test Acc | Valid Acc (Initial→Final) | Valid Error (Initial→Final) | Epochs |\n"
        markdown += "|---------------|------------|-----|--------|----------|---------------------------|----------------------------|--------|\n"

        for result in successful_results:
            hidden = str(result["hidden_neurons"])
            activation = result["activation"]
            eta = result["eta"]
            method = result["method"]
            if method == "k-fold" and "k_folds" in result:
                method += f" (k={result['k_folds']})"
            test_acc = f"{result['test_accuracy']*100:.2f}%"
            
            # Validation accuracy progression
            init_valid_acc = result.get('initial_valid_accuracy', 0)
            final_valid_acc = result.get('final_valid_accuracy', 0)
            valid_acc_str = f"{init_valid_acc*100:.2f}%→{final_valid_acc*100:.2f}%"
            
            # Validation error progression
            init_valid_err = result.get('initial_valid_error', 0)
            final_valid_err = result.get('final_valid_error', 0)
            valid_err_str = f"{init_valid_err:.4f}→{final_valid_err:.4f}"
            
            epochs_trained = result.get('epochs_trained', 'N/A')

            markdown += f"| {hidden} | {activation} | {eta} | {method} | {test_acc} | {valid_acc_str} | {valid_err_str} | {epochs_trained} |\n"

        # Summary Statistics
        accuracies = [r["test_accuracy"] for r in successful_results]
        markdown += f"\n---\n\n## Summary Statistics\n\n"
        markdown += f"- **Total Successful Experiments**: {len(successful_results)}\n"
        markdown += f"- **Best Accuracy**: {max(accuracies)*100:.2f}%\n"
        markdown += f"- **Worst Accuracy**: {min(accuracies)*100:.2f}%\n"
        markdown += f"- **Mean Accuracy**: {np.mean(accuracies)*100:.2f}%\n"
        markdown += f"- **Standard Deviation**: {np.std(accuracies)*100:.2f}%\n"

        # Best Configuration
        best_idx = np.argmax(accuracies)
        best_config = successful_results[best_idx]
        markdown += f"\n---\n\n## Best Configuration\n\n"
        markdown += f"- **Hidden Layers**: {best_config['hidden_neurons']}\n"
        markdown += f"- **Activation Function**: {best_config['activation']}\n"
        markdown += f"- **Learning Rate (eta)**: {best_config['eta']}\n"
        markdown += f"- **Method**: {best_config['method']}\n"
        markdown += f"- **Test Accuracy**: {best_config['test_accuracy']*100:.2f}%\n"
        markdown += f"- **Epochs**: {best_config['epoch_number']}\n"
        markdown += f"- **Patience**: {best_config['patience']}\n"

        # Analyze by activation function
        markdown += ResultsManager._analyze_by_activation(successful_results)

        # Analyze by architecture
        markdown += ResultsManager._analyze_by_architecture(successful_results)

        # Analyze by learning rate
        markdown += ResultsManager._analyze_by_learning_rate(successful_results)

        with open(filename, "w", encoding="utf-8") as f:
            f.write(markdown)

        print(f"Markdown report saved to {filename}")

    @staticmethod
    def _analyze_by_activation(results: list) -> str:
        """Analyze results by activation function."""
        markdown = f"\n---\n\n## Analysis by Activation Function\n\n"
        activations = {}
        for result in results:
            activation = result["activation"]
            if activation not in activations:
                activations[activation] = []
            activations[activation].append(result["test_accuracy"])

        markdown += "| Activation Function | Best Accuracy | Mean Accuracy | Experiments |\n"
        markdown += "|---------------------|---------------|---------------|-------------|\n"
        for activation, accs in sorted(activations.items()):
            best = max(accs) * 100
            mean = np.mean(accs) * 100
            count = len(accs)
            markdown += f"| {activation} | {best:.2f}% | {mean:.2f}% | {count} |\n"

        return markdown

    @staticmethod
    def _analyze_by_architecture(results: list) -> str:
        """Analyze results by architecture."""
        markdown = f"\n---\n\n## Analysis by Architecture\n\n"
        architectures = {}
        for result in results:
            arch = str(result["hidden_neurons"])
            if arch not in architectures:
                architectures[arch] = []
            architectures[arch].append(result["test_accuracy"])

        markdown += "| Architecture | Best Accuracy | Mean Accuracy | Experiments |\n"
        markdown += "|--------------|---------------|---------------|-------------|\n"
        for arch, accs in sorted(architectures.items()):
            best = max(accs) * 100
            mean = np.mean(accs) * 100
            count = len(accs)
            markdown += f"| {arch} | {best:.2f}% | {mean:.2f}% | {count} |\n"

        return markdown

    @staticmethod
    def _analyze_by_learning_rate(results: list) -> str:
        """Analyze results by learning rate."""
        markdown = f"\n---\n\n## Analysis by Learning Rate\n\n"
        learning_rates = {}
        for result in results:
            eta = result["eta"]
            if eta not in learning_rates:
                learning_rates[eta] = []
            learning_rates[eta].append(result["test_accuracy"])

        markdown += "| Learning Rate (eta) | Best Accuracy | Mean Accuracy | Experiments |\n"
        markdown += "|--------------------|---------------|---------------|-------------|\n"
        for eta, accs in sorted(learning_rates.items()):
            best = max(accs) * 100
            mean = np.mean(accs) * 100
            count = len(accs)
            markdown += f"| {eta} | {best:.2f}% | {mean:.2f}% | {count} |\n"

        return markdown

    @staticmethod
    def print_summary(results: list):
        """Print a summary of experiment results."""
        successful_results = [r for r in results if r["status"] == "success"]

        if not successful_results:
            print("\nWarning: No successful experiments completed.")
            return

        print("\n" + "=" * 80)
        print("EXPERIMENTS COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print(f"Total experiments run: {len(results)}")
        print(f"Successful experiments: {len(successful_results)}")

        accuracies = [r["test_accuracy"] for r in successful_results]
        print(f"\nBest accuracy: {max(accuracies)*100:.2f}%")
        print(f"Average accuracy: {np.mean(accuracies)*100:.2f}%")
        print(f"Worst accuracy: {min(accuracies)*100:.2f}%")

        # Find and display best model
        best_idx = np.argmax(accuracies)
        best_result = successful_results[best_idx]
        print(f"\n{'='*80}")
        print("BEST CONFIGURATION")
        print(f"{'='*80}")
        print(f"Hidden layers: {best_result['hidden_neurons']}")
        print(f"Activation function: {best_result['activation']}")
        print(f"Learning rate: {best_result['eta']}")
        print(f"Method: {best_result['method']}")
        print(f"Test accuracy: {best_result['test_accuracy']*100:.2f}%")
        print(f"{'='*80}\n")