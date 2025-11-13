import numpy as np
import matplotlib.pyplot as plt
from network import Network
from typing import Optional


class Visualizer:
    """Handle visualization of predictions and results."""

    @staticmethod
    def show_prediction(network: Network, X_test: np.ndarray, Y_test: np.ndarray, random_seed: Optional[int] = None) -> None:
        """Visualize a random test image with its prediction."""
        if random_seed is not None:
            np.random.seed(random_seed)

        random_index = np.random.randint(0, X_test.shape[0])
        img_test = X_test[random_index]

        # Get the true and predicted labels
        true_label = np.argmax(Y_test[random_index])
        predicted_label = network.predict_single(img_test)

        print(f"\nTrue label: {true_label}")
        print(f"Predicted label: {predicted_label}")

        # Visualize the test image
        img_test = img_test.reshape(28, 28)
        plt.imshow(img_test, cmap="gray")
        plt.title(f"True label: {true_label} | Predicted: {predicted_label}")
        plt.axis("off")
        plt.show()

    @staticmethod
    def show_multiple_predictions(
        network: Network, X_test: np.ndarray, Y_test: np.ndarray, num_samples: int = 10
    ) -> None:
        """Visualize multiple test predictions in a grid."""
        indices = np.random.choice(X_test.shape[0], num_samples, replace=False)

        rows = int(np.ceil(num_samples / 5))
        cols = min(5, num_samples)

        fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))
        axes = axes.flatten() if num_samples > 1 else [axes]

        for idx, ax in zip(indices, axes):
            img_test = X_test[idx].reshape(28, 28)
            true_label = np.argmax(Y_test[idx])
            predicted_label = network.predict_single(X_test[idx])

            ax.imshow(img_test, cmap="gray")
            color = "green" if true_label == predicted_label else "red"
            ax.set_title(f"True: {true_label} | Pred: {predicted_label}", color=color)
            ax.axis("off")

        # Hide extra subplots
        for ax in axes[num_samples:]:
            ax.axis("off")

        plt.tight_layout()
        plt.show()