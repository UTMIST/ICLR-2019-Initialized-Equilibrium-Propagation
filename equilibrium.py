"""
Defines a model object.
"""
from typing import List, Tuple
import numpy as np


class Equilibrium:
    """
    A neural net to be trained using equilibrium propagation.
    """

    def __init__(self, shape: Tuple[int]):
        """
        Shape = [D1, D2, ..., DN] where Di the dimensionality of the ith layer in a FCN
        """
        self.shape = shape
        self.state = [np.zeros(i) for i in self.shape[1:]]
        self.weights = self.init_weights(self.shape)
        self.bias = [np.zeros(i) for i in self.shape]

    @staticmethod
    def get_test_network():
        """
        Get a simple neural network to test with having shape [3, 4, 4, 3]

        >>> testnet = Equilibrium.get_test_network()
        >>> testnet.shape
        (3, 4, 4, 3)
        >>> testnet.state
        [array([0., 0., 0., 0.]), array([0., 0., 0., 0.]), array([0., 0., 0.])]
        >>> len(testnet.weights)
        3
        >>> testnet.weights[0].shape
        (3, 4)
        >>> testnet.weights[1].shape
        (4, 4)
        >>> testnet.weights[2].shape
        (4, 3)
        >>> [len(b) for b in testnet.bias] == [3, 4, 4, 3]
        True
        """
        return Equilibrium((3, 4, 4, 3))

    @staticmethod
    def rho(v):
        """
        The activation function to be used, here a hard sigmoid

        >>> v = [1, 2, -1, -2, 0.5, -0.5]
        >>> Equilibrium.rho(v)
        array([1. , 1. , 0. , 0. , 0.5, 0. ])
        """
        t = np.clip(v, 0, 1)
        return t

    @staticmethod
    def rhoprime(v):
        """
        The gradient of the activation function to be used, here a hard sigmoid

        >>> v = [0.9, 2, -1, -2, 0.5, -0.5]
        >>> Equilibrium.rhoprime(v)
        array([1, 0, 0, 0, 1, 0])
        """
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
            rng = np.random.RandomState()
            return np.asarray(rng.uniform(
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
        return self.state

    def positive_phase(self, x, y, t_plus: int, epsilon: float, beta: float):
        """
        Positive phase training.
        """
        for _ in range(t_plus):
            self.state -= epsilon * self.clamped_energy_grad(x, y, beta)
        return self.state

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
        # non-input biases
        for j in range(len(activations)):
            state = activations[j]
            sum_ += np.dot(state, self.bias[j + 1])
        # input weights
        sum_ += np.dot(np.dot(self.weights[0], x), activations[0])

        return sum_

    def energy_grad_state(self, x):
        """
        Returns the gradient of the energy function evaluated at the current state.
        """
        # TODO: work out by hand what the gradients should be (refer to original code)
        # self.state = [np.zeros(shape[i]) for i in range(1, len(shape))]
        activations_prime = [self.rhoprime(i) for i in self.state]
        activations = [self.rho(i) for i in self.state]

        state_grad = [np.zeros(i) for i in self.shape[1:]]

        for layer_index in range(len(state_grad)):
            activated_prime_state = activations_prime[layer_index]
            weights = self.weights[layer_index]
            bias = self.bias[layer_index]

            if layer_index == 0:  # 2nd layer. Consider only 3rd layer.
                next_activated_state = activations[layer_index + 1]
                state_grad[layer_index] = (-self.state[layer_index] +
                                           np.multiply(activated_prime_state,
                                                       (bias +
                                                        np.matmul(np.transpose(weights), next_activated_state) +
                                                        np.matmul(weights, x))))

            elif layer_index == len(state_grad) - 1:  # Last layer. Consider only before-last layer.
                prev_activated_state = activations[layer_index - 1]
                prev_weights = self.weights[layer_index-1]
                state_grad[layer_index] = (-self.state[layer_index] +
                                           np.multiply(activated_prime_state,
                                                       (bias +
                                                        np.matmul(prev_weights, prev_activated_state))))  # Don't use inputs
            else:
                next_activated_state = activations[layer_index + 1]
                prev_activated_state = activations[layer_index - 1]
                prev_weights = self.weights[layer_index-1]

                state_grad[layer_index] = (-self.state[layer_index] +
                                           np.multiply(activated_prime_state,
                                                       (bias +
                                                        np.matmul(np.transpose(weights), next_activated_state) +
                                                        np.matmul(prev_weights, prev_activated_state))))
        return state_grad

    def update_weights(self, beta, eta, s_pos, s_neg, x):
        """
        Returns the gradient of the energy function evaluated at the current state.
        """
        act_neg = [self.rho(i) for i in s_neg]
        act_pos = [self.rho(j) for j in s_pos]
        # get gradient
        weight_shape = zip(self.shape[:-1], self.shape[1:])
        grad_weight = [np.zeros(shape) for shape in weight_shape]
        grad_bias = [np.zeros(i) for i in self.shape]
        # loop over non-input weight layers
        # off by 1 in act_pos/act_neg??
        for i in range(1, len(act_neg) - 1):
            grad_weight[i] = (eta/beta) * (np.dot(act_pos[i-1],
                                                 act_pos[i].T) - np.dot(act_neg[i-1], act_neg[i].T))
            # bias gradient
            grad_bias[i] = (eta/beta) * (act_pos[i] - act_neg[i])
            # input??? null
        grad_weight[0] = (eta/beta) * (np.dot(x, act_pos[0].T) - np.dot(x, act_neg[0].T))

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


if __name__ == "__main__":
    import doctest
    doctest.testmod()
