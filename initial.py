"""
The initializer net.
"""
from typing import List
import numpy as np


class Initial:
    """
    A feedforward neural network trained to initalize the state of the equilibriating net for training via equilibrium
    propagation.
    """

    def __init__(self, shape: List[int]):
        """
        Shape = [D1, D2, ..., DN] where Di the dimensionality of the ith layer in a FCN
        """
        self.shape = shape
        self.state = [np.zeros(shape[i]) for i in range(1, len(self.shape))]
        self.weights = self.init_weights(self.shape)
        self.bias = [np.zeros(shape[i]) for i in range(0, len(self.shape))]

    @staticmethod
    def h(v):
        """
        The activation function to be used, here a hard sigmoid

        >>> v = [1, 2, -1, -2, 0.5, -0.5]
        >>> Initial.rho(v)
        array([1. , 1. , 0. , 0. , 0.5, 0. ])
        """
        # TODO: add leak
        t = np.clip(v, 0, 1)
        return t

    @staticmethod
    def hprime(v):
        """
        The gradient of the activation function to be used, here a hard sigmoid

        >>> v = [0.9, 2, -1, -2, 0.5, -0.5]
        >>> Equilibrium.rhoprime(v)
        array([1, 0, 0, 0, 1, 0])
        """
        # TODO: add leak
        v = np.asarray(v)
        return ((v >= 0) & (v <= 1)).astype(int)

    @staticmethod
    def init_weights(shape: List[int]):
        """
        Initialize the weights according to Glorot/Bengio initialization.

        >>> w = Equilibrium.init_weights([1, 2, 3])
        >>> len(w)
        2
        >>> w[0].shape
        (1, 2)
        >>> (w[0] >= -np.sqrt(2)).all() and (w[0] <= np.sqrt(2)).all()
        True
        >>> w[1].shape
        (2, 3)
        >>> (w[1] >= -np.sqrt(6 / 5)).all() and (w[1] <= np.sqrt(6 / 5)).all()
        True
        """

        def get_initialized_layer(n_in, n_out):
            """
            Perform Glorot/Bengio initialization of a single layer of the network
            with input dimension n_in and output dimension n_out

            >>> w = get_initialized_layer(3, 4)
            >>> w.shape
            (3, 4)
            >>> (w >= -np.sqrt(6 / 7)).all() and (w <= np.sqrt(6 / 7)).all()
            True
            """
            # TODO:  use rng?
            rng = np.random.RandomState()
            return np.asarray(np.random.uniform(
                -np.sqrt(6 / (n_in + n_out)),
                np.sqrt(6 / (n_in + n_out)),
                (n_in, n_out)
            ))

        weight_shape = zip(shape[:-1], shape[1:])
        return [get_initialized_layer(n_in, n_out)
                for n_in, n_out in weight_shape]

    def feedforward(self, x):
        """
        Feedforward to get initializing state for equlibrium net.
        """
        self.state[0] = Initial.h(np.dot(self.weights[0], x) + self.bias[0])
        for i in range(1,  len(self.shape)):
            self.state[i] = Initial.h(np.dot(self.weights[i], self.state[i - 1]) + self.bias[i - 1])
        return self.state

    def update_weights(self, eta, s_neg, x):
        """
        Returns the gradient of the energy function evaluated at the current state.
        """
        # outer product
