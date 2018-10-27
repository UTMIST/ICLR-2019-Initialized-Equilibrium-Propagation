import numpy as np

def initialize_feedforward_parameters(model):
    """Initialize and return phi (parameters of the feedforward NN)"""
    phi=0
    return phi

def initialize_equilibrium_parameters(model):
    """Initialize and return theta (parameters of the equilibrium NN)"""
    theta = 0

    return theta

def train(hyperparams: dict):

    while True:
        #Sample minibatch

        for t in hyperparams["t-"]:
            pass #Neg phase

        for t in hyperparams["t+"]:
            pass #Pos phase

        #theta = 

        pass


if __name__ == "__main__":
    hyperparams = {
        "epsilon": 0.01,
        "eta": 0.1,
        "t+": 1,
        "t-": 1
    }
