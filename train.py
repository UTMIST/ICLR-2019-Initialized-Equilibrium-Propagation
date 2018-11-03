import numpy as np

from data import load_data

SEED = 0


def initialize_feedforward_parameters(model):
    """Initialize and return phi (parameters of the feedforward NN)"""
    phi = 0
    return phi


def initialize_equilibrium_parameters(model):
    """Initialize and return theta (parameters of the equilibrium NN)"""
    theta = 0

    return theta


def train(hyperparams: dict):
    """
    Train the model with <hyperparams>.
    """
    train, val, test = load_data(0.8)  # TODO: train/val split ratio?
    while True:
        # Sample minibatch

        for t in hyperparams["t-"]:
            pass  # Neg phase

        for t in hyperparams["t+"]:
            pass  # Pos phase

        #theta =

        pass


if __name__ == "__main__":
    hyperparams = {
        "epsilon": 0.01,        # step size
        "eta": 0.1,             # learning rate
        "t+": 1,                # # of pos phase steps
        "t-": 1,                # # of neg phase steps
        "alpha": []             # architecture, specified as sizes of hidden layers
    }
    train(hyperparams)
