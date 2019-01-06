"""
For evaluating the model (e.g. accuracy).
"""

import numpy as np


def accuracy(model, x, y, hyperparams):
    """
    Evaluate accuracy of model on dataset with examples x and targets y.
    """
    predictions = model.negative_phase(x, hyperparams["t-"], hyperparams["epsilon"])[-1]
    return np.sum(np.argmax(predictions, axis=0) == np.argmax(y, axis=0)) / predictions.shape[1]
