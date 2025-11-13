import numpy as np


def _soft_max(Z):
    """Softmax function."""
    # Z - Z.max(0) for avoid overflow
    exp_Z = np.exp(Z - Z.max(axis=1, keepdims=True))
    Z = exp_Z / exp_Z.sum(axis=1, keepdims=True)
    return Z


# Cross-entropy error function
def cross_entropy(Z_label, Y_label, derivative=False):
    """Cross-entropy error function."""
    # Z is a matrix with probability
    Z = _soft_max(Z_label)
    # Clip to avoid log(0) - more robust than adding epsilon
    Z = np.clip(Z, 1e-15, 1 - 1e-15)
    if not derivative:
        # Normalize by number of samples for consistent error values
        return -(Y_label * np.log(Z)).sum() / Y_label.shape[0]
    # Derivative of error function respect weights
    return Z - Y_label
