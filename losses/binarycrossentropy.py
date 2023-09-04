import numpy as np

class BinaryCrossEntropy:
    def __init__(self):
        pass

    def compute(self, y_hat: np.ndarray, y: np.ndarray) -> float:
        """
        Computes the binary cross entropy loss.
            args:
                y: true labels (n_classes, batch_size)
                y_hat: predicted labels (n_classes, batch_size)
            returns:
                binary cross entropy loss
        """
        batch_size = y.shape[1]
        cost = -1/batch_size * np.sum(np.multiply(y, np.log(y_hat)) + np.multiply(1-y, np.log(1-y_hat)))
        return np.squeeze(cost)

    def backward(self, y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Computes the derivative of the binary cross entropy loss.
            args:
                y: true labels (n_classes, batch_size)
                y_hat: predicted labels (n_classes, batch_size)
            returns:
                derivative of the binary cross entropy loss
        """
        # hint: use the np.divide function
        return np.divide(y_hat - y, np.multiply(y_hat, 1-y_hat))

