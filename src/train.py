"""
For training the model
"""

import numpy as np
import torch
from equilibrium import EquilibriumNet
from evaluate import accuracy

from data import load_data

SEED = 0


def train(hyperparams: dict):
    """
    Train the model with <hyperparams>.
    """

    train, val, test = load_data(0.8)  # TODO: train/val split ratio?

    train = test

    model = EquilibriumNet(*hyperparams["alpha"])


    # get number of examples
    train_size = train[0].shape[1]
    for epoch in range(hyperparams["epochs"]):

        # shuffle train set
        shuffled_indices = [i for i in range(train_size)]
        np.random.shuffle(shuffled_indices)
        for minibatch in range(int(np.ceil(train_size/model.minibatch_size))):
            x_sample = train[0][:, shuffled_indices[minibatch * hyperparams["alpha"][-1]:
                                                    (minibatch + 1) * hyperparams["alpha"][-1]]]
            y_sample = train[1][:, shuffled_indices[minibatch * hyperparams["alpha"][-1]:
                                                    (minibatch + 1) * hyperparams["alpha"][-1]]]
            x_sample = torch.tensor(x_sample.T).float()
            y_sample = torch.tensor(y_sample).float()
            beta = (2 * np.random.randint(0, 2) - 1) * hyperparams["beta"]  # random sign
            s_neg = model.negative_phase(x_sample, hyperparams["t-"], hyperparams["epsilon"])
            s_pos = model.positive_phase(x_sample, y_sample, hyperparams["t+"], hyperparams["epsilon"], beta)

            model.update_weights(beta, hyperparams["etas"], s_pos, s_neg, x_sample)

            if minibatch % 100 == 0:
                print("Epoch: %d, Minibatch: %d" % (epoch, minibatch))

        total_accuracy = 0
        num_accuracies = 0
        for minibatch in range(int(np.ceil(train_size/model.minibatch_size))):
            x_sample = train[0][:, shuffled_indices[minibatch * hyperparams["alpha"][-1]:
                                                    (minibatch + 1) * hyperparams["alpha"][-1]]]
            y_sample = train[1][:, shuffled_indices[minibatch * hyperparams["alpha"][-1]:
                                                    (minibatch + 1) * hyperparams["alpha"][-1]]]
            x_sample = torch.tensor(x_sample.T).float()
            y_sample = torch.tensor(y_sample).float()
            total_accuracy += accuracy(model, x_sample, y_sample, hyperparams)
            num_accuracies += 1
        print("Epoch: %d Accuracy: %.2f" % (epoch, total_accuracy / num_accuracies))


if __name__ == "__main__":
    hyperparams = {
        "epsilon": 0.5,                     # step size
        "beta": 0.5,                        # clamping factor
        "etas": [0.1, 0.05],                # learning rate  # TODO: should this be lr = eta/beta?
        "t+": 4,                            # # of pos phase steps
        "t-": 20,                           # # of neg phase steps
        "alpha": (784, [500], 10, 20),      # architecture, specified as sizes of hidden layers
        "epochs": 25
    }
    train(hyperparams)
