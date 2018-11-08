"""
Defines a model object.
"""
from typing import List

import numpy as np


class Equilibriating:
    """
    A neural net to be trained using equilibrium propagation.
    """

    def __init__(self, shape: List[int]):
        """
        Shape = [D1, D2, ..., DN] where Di the dimensionality of the ith layer in a FCN
        """
        self.input_dim = shape[0]
        self.state = [np.zeros(shape[i]) for i in range(1, len(shape))]
        self.weights = self.init_weights(shape)
        self.bias = [np.zeros(shape[i]) for i in range(1, len(shape))]

    @staticmethod
    def rho(v):
        """
        The activation function to be used, here a hard sigmoid

        >>> v = [1, 2, -1, -2, 0.5, -0.5]
        >>> Equilibriating.rho(v)
        array([1. , 1. , 0. , 0. , 0.5, 0. ])
        """
        t = np.clip(v, 0, 1)
        return t

    @staticmethod
    def rhoprime(v):
        """
        The gradient of the activation function to be used, here a hard sigmoid

        >>> v = [0.9, 2, -1, -2, 0.5, -0.5]
        >>> Equilibriating.rhoprime(v)
        array([1, 0, 0, 0, 1, 0])
        """
        v = np.asarray(v)
        return ((v >= 0) & (v <= 1)).astype(int)

    @staticmethod
    def init_weights(shape: List[int]):
        """
        Initialize the weights according to Glorot/Bengio initialization.

        >>> w = Equilibriating.init_weights([1, 2, 3])
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
            rng = np.random.RandomState()
            return np.asarray(np.random.uniform(
                -np.sqrt(6 / (n_in + n_out)),
                np.sqrt(6 / (n_in + n_out)),
                (n_in, n_out)
            ))

        weight_shape = zip(shape[:-1], shape[1:])
        return [get_initialized_layer(n_in, n_out)
                for n_in, n_out in weight_shape]

    def outputs(self):
        """
        Returns the output of the net.
        """
        return self.state[-1]

    def negative_phase(self, x, t_minus: int, epsilon: float):
        """
        Negative phase training.
        """
        for _ in range(t_minus):
            self.state -= epsilon * self.energy_grad_state(x)

    def positive_phase(self, x, y, t_plus: int, epsilon: float, beta: float):
        """
        Positive phase training.
        """
        for _ in range(t_plus):
            self.state -= epsilon * self.clamped_energy_grad(x, y, beta)

    def energy(self, x):
        """
        Returns the energy of the net.
        """
        # TODO: add unit test
        activations = [self.rho(i) for i in self.state]
        sum_ = sum([np.sum(state ** 2) for state in self.state]) / 2  # magnitude of state
        for i in range(len(activations) - 1):
            state = activations[i]
            next_ = activations[i + 1]
            sum_ += 2 * np.dot(np.dot(self.weights[i], state), next_)
        for j in range(len(activations)):
            state = activations[j]
            next_ = activations[j + 1]
            sum_ += np.dot(state, next_)
        sum_ += np.dot(np.dot(self.weights[0], x), activations[0])  # add input

        return sum_

    def energy_grad_state(self, x):
        """
        Returns the gradient of the energy function evaluated at the current state.
        """
        # TODO: work out by hand what the gradients should be (refer to original code)
        return 0

    def energy_grad_weights(self, x):
        """
        Returns the gradient of the energy function evaluated at the current state.
        """
        # TODO: work out by hand what the gradients should be (refer to original code)
        return 0

    def clamped_energy_grad(self, x, y, beta):
        """
        Returns the gradient of the clamped energy function evaluated at the current state and target.
        """
        return self.energy_grad_state(x) + 2 * beta * (y - self.outputs())

    def _energy_grad_state_check(self):
        """

        :return:
        :rtype:
        """
        pass

    def _energy_grad_weight_check(self):
        """

        :return:
        :rtype:
        """
        pass


class Initializer:
    """
    A feedforward neural network trained to initalize the state of the equilibriating net for training via equilibrium
    propagation.
    """

    def __init__(self):
        pass

    def evaluate(self):
        """
        Returns the output of the net.
        """
        return 0


if __name__ == "__main__":
    import doctest
    doctest.testmod()
