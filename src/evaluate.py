"""
For evaluating the model (e.g. accuracy).
"""

import numpy as np


def accuracy(model, x, y, hyperparams):
    """
    Evaluate accuracy of model on dataset with examples x and targets y.
    """
    predictions = np.array(model.negative_phase(x, hyperparams["t-"], hyperparams["epsilon"])[model.partial_sums[-2]:])
    return np.sum(np.argmax(predictions, axis=0) == np.argmax(np.array(y), axis=0)) / predictions.shape[1]
