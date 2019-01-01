"""
Defines a model object.
"""
from typing import Tuple

import numpy as np


class Equilibrium:
    """
    A neural net to be trained using equilibrium propagation.
    """

    def __init__(self, shape: Tuple[int, ...]):
        """
        Shape = [D1, D2, ..., DN] where Di the dimensionality of the ith layer in a FCN

        Note: may need to change initialization of biases since input shouldn't have bias
        """
        self.shape = shape
        self.state = [np.zeros(i) for i in self.shape[1:]]
        self.weights = self.init_weights(self.shape)
        self.bias = [np.zeros(i) for i in self.shape]

    @staticmethod
    def test_shapes():
        """
        Get a simple neural network with layer sizes (3, 4, 4, 3) to test shapes of weights, biases, state.

        >>> testnet = Equilibrium.test_shapes()
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
    def test_energy():
        """
        Get a simple neural network with layer sizes (3, 5, 3) to test energy calculations.

        >>> testnet = Equilibrium.test_energy()
        >>> x = np.zeros(3)
        >>> testnet.energy(x)
        0.5
        >>> x = np.array([i for i in range(3)])
        >>> testnet.energy(x)
        -70.5
        """
        testnet = Equilibrium((3, 5, 3))
        testnet.weights[0] = np.array([[i for i in range(15)]]).reshape(3, 5)
        testnet.weights[1] = np.array([[i for i in range(15)]]).reshape(5, 3)
        testnet.bias[0] = np.array([i for i in range(3)])
        testnet.bias[1] = np.array([i for i in range(5)])
        testnet.bias[2] = np.array([i for i in range(3)])
        testnet.state[0] = np.array([i - 2 for i in range(5)])
        testnet.state[1] = np.array([i - 2 for i in range(3)])

        return testnet

    @staticmethod
    def test_update_weights():
        """
        # Test update_weights method

        # >>> testnet = Equilibrium.test_update_weights()
        # >>> x = np.array([i for i in range(3)])
        # >>> s_pos = [np.zeros(i) for i in testnet.shape[1:]]
        # >>> s_neg = [np.zeros(i) for i in testnet.shape[1:]]
        # >>> beta, eta = 1, 1
        # >>> testnet.update_weights(beta, eta, s_pos, s_neg, x)
        """
        testnet = Equilibrium((3, 4, 5, 3))
        testnet.weights[0] = np.array([[i for i in range(12)]], dtype = np.float64).reshape(3, 4)
        testnet.weights[1] = np.array([[i for i in range(20)]], dtype = np.float64).reshape(4, 5)
        testnet.weights[2] = np.array([[i for i in range(15)]], dtype = np.float64).reshape(5, 3)
        testnet.bias[0] = np.array([i for i in range(3)], dtype=np.float64)
        testnet.bias[1] = np.array([i for i in range(4)], dtype=np.float64)
        testnet.bias[2] = np.array([i for i in range(5)], dtype=np.float64)
        testnet.bias[3] = np.array([i for i in range(3)], dtype=np.float64)

        x = np.array([i for i in range(3)])
        s_pos = [np.zeros(i) for i in testnet.shape[1:]]
        s_neg = [np.zeros(i) for i in testnet.shape[1:]]
        beta, eta = 1, [1]
        testnet.update_weights(beta, eta, s_pos, s_neg, x)

        return testnet

    @staticmethod
    def test_grad_check_state():
        """
        """
        testnet = Equilibrium((3, 5, 3))
        testnet.weights[0] = np.array([[i for i in range(15)]]).reshape(3, 5)
        testnet.weights[1] = np.array([[i for i in range(15)]]).reshape(5, 3)
        testnet.bias[0] = np.array([i for i in range(3)])
        testnet.bias[1] = np.array([i for i in range(5)])
        testnet.bias[2] = np.array([i for i in range(3)])
        testnet.state[0] = np.array([(2*i+1)/10 for i in range(5)])
        testnet.state[1] = np.array([(2*i+1)/10 for i in range(3)])
        testnet._energy_grad_state_check()

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
    def init_weights(shape: Tuple[int, ...]):
        """
        Initialize the weights according to Glorot/Bengio initialization.

        >>> w = Equilibrium.init_weights((1, 2, 3))
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
        batch_size = x.shape[-1]
        state = [np.zeros((i, batch_size)) for i in self.shape[1:]]
        for _ in range(t_minus):
            grad = self.energy_grad_state(x)
            for i in range(len(self.state)):
                state[i] -= epsilon * grad[i]
        return state

    def positive_phase(self, x, y, t_plus: int, epsilon: float, beta: float):
        """
        Positive phase training.
        """
        batch_size = x.shape[-1]
        state = [np.zeros((i, batch_size)) for i in self.shape[1:]]
        for _ in range(t_plus):
            grad = self.clamped_energy_grad(state, x, y, beta)
            for i in range(len(self.state)):
                state[i] -= epsilon * grad[i]
        return state

    def energy(self, x):
        """
        Returns the energy of the net. (Equation 1).
        If x is a minibatch of examples, return the average of the energy of the net on each of these examples.
        """
        try:
            batch_size = x.shape[1]
        except IndexError:
            batch_size = 1
        activated_states = [self.rho(i) for i in self.state]
        state_norm = sum([np.sum(state ** 2) for state in self.state]) / 2
        total_energy = state_norm

        for state, next_state, weights in zip(activated_states[:-1], activated_states[1:], self.weights[1:]):
            total_energy -= 2 * np.dot(np.dot(weights.T, state), next_state)

        # non-input biases
        for state, bias in zip(activated_states, self.bias[1:]):
            total_energy -= np.dot(state, bias)

        # input weights
        input_weights = np.dot(np.dot(self.weights[0].T, x).T, activated_states[0])

        return np.sum(total_energy - input_weights) / batch_size

    def calc_grad(self, curr_state, activated_prime_state, bias, prev_activated_state, prev_weights,
                  next_weights, next_activated_state):
        """Returns the gradient for one state layer

        >>> testnet = Equilibrium.test_shapes()
        >>> curr_state = np.array([9, 9, 9, 9])
        >>> act_pr_st = np.array([4, 5, 6, 7])
        >>> bias = np.array([1, 1, 1, 1])
        >>> prev_activated_state = np.array([1,2,3])
        >>> prev_w = np.zeros((3,4))
        >>> next_w = np.eye(4)
        >>> next_act_st = np.array([7,8,9,10])
        >>> testnet.calc_grad(curr_state, act_pr_st, bias, prev_activated_state, prev_w, next_w, next_act_st)
        array([23., 36., 51., 68.])
        """
        first_sum = np.matmul(next_weights, next_activated_state)
        second_sum = np.matmul(prev_weights.T, prev_activated_state)
        big_sum = bias + first_sum + second_sum

        return -curr_state + np.multiply(activated_prime_state, big_sum)

    def energy_grad_state(self, x):
        """
        Returns the gradient of the energy function evaluated at the current state.

        >>> testnet = Equilibrium.test_shapes()
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
        >>> x = np.array([1,1,2])
        """
        # >>> testnet.energy_grad_state(x)
        # [array([ 0.13709111, -1.72075492,  1.57004124, -0.22437192]), array([0., 0., 0., 0.]), array([0., 0., 0.])]
        # Result depends on random numbers.
        # But the sizes are right and the last 2 gradient vectors are all 0's, which is correct
        # (next_activated_state are 0).

        batch_size = x.shape[-1]
        activations_prime = [self.rhoprime(layer) for layer in self.state]
        activations = [self.rho(layer) for layer in self.state]
        state_grad = [np.zeros((i, batch_size)) for i in self.shape[1:]]
        last_index = len(self.state) - 1
        size_last_layer = self.shape[-1]

        for layer_index in range(len(self.state)):
            curr_state = np.repeat(np.expand_dims(self.state[layer_index], axis=1), batch_size, axis=1)
            activated_prime_state = np.repeat(np.expand_dims(activations_prime[layer_index], axis=1), batch_size, axis=1)
            prev_weights = self.weights[layer_index]
            bias = np.expand_dims(self.bias[layer_index + 1], axis=1)

            prev_activated_state = x if layer_index == 0 \
                else np.repeat(np.expand_dims(activations[layer_index - 1], axis=1), batch_size, axis=1)

            next_activated_state = np.zeros((size_last_layer, batch_size)) if layer_index == last_index \
                else np.repeat(np.expand_dims(activations[layer_index + 1], axis=1), batch_size, axis=1)

            next_weights = np.zeros((size_last_layer, size_last_layer)) if layer_index == last_index \
                else self.weights[layer_index + 1]

            # print(activated_prime_state, bias, prev_activated_state, next_activated_state, next_weights)

            state_grad[layer_index] = self.calc_grad(curr_state, activated_prime_state, bias, prev_activated_state,
                                                     prev_weights, next_weights, next_activated_state)

        return state_grad

    def update_weights(self, beta, etas, s_pos, s_neg, x):
        """
        Updates the weights by the gradient of the energy function evaluated at the current state.
        beta: clamping factor
        etas: learning rates (one for each layer)
        s_pos: state from positive phase training
        s_neg: state from negative phase training
        x: a batch of inputs
        """
        batch_size = x.shape[-1]
        act_neg = [self.rho(i) for i in s_neg]
        act_pos = [self.rho(j) for j in s_pos]
        # loop over non-input weight layers
        for i in range(1, len(self.state) - 1):
            self.weights[i] -= (etas[i]/beta) * \
                               (np.dot(act_pos[i-1], act_pos[i].T) - np.dot(act_neg[i-1], act_neg[i].T)) / batch_size
            # bias gradient
            self.bias[i] -= (etas[i]/beta) * np.sum(act_pos[i - 1] - act_neg[i - 1], axis=1) / batch_size

        self.weights[0] -= (etas[0]/beta) * (np.dot(x, act_pos[0].T) - np.dot(x, act_neg[0].T)) / batch_size


    # TODO: feedforward prediction to help feedforward neurons learn a mapping from previous layer's activations to targets given by this network (formula 10)
    def closer_energy(self, lamda, state, x):
        """
        lambda: hyperparameter that brings fixed-points of equilibrating network closer to states of forward pass
        x: input neurons
        state: states achieved by neurons in feedforward
        """

    def clamped_energy_grad(self, state, x, y, beta):
        """
        Returns the gradient of the clamped energy function evaluated at the current state and target.
        """
        grad = self.energy_grad_state(x)
        clamp = 2 * beta * (state[-1] - y)
        clamped_grad = grad
        clamped_grad[-1] += clamp
        return clamped_grad

    def _energy_grad_state_check(self, dh=10e-10):
        """
        Verify that our energy function states are close to gradient
        """
        size = self.shape[0]
        x = np.ones((size, 1))
        gradient = self.energy_grad_state(x)
        for layer in range(len(self.state)):
            for neuron in range(len(self.state[layer])):
                self.state[layer][neuron] += dh
                f_plus = self.energy(x)
                self.state[layer][neuron] -= 2*dh
                f_neg = self.energy(x)
                grad_check = (f_plus - f_neg) / 2*dh
                error = Equilibrium.calc_relative_error(gradient[layer][neuron], grad_check)
                if error >= 10e-6:
                    stringg = "layer: {} neuron: {} error: {} grad check: {} true grad: {}".format(
                        layer, neuron, error, grad_check, gradient[layer][neuron])
                    print(stringg)

    @staticmethod
    def calc_relative_error(a, b):
        """
        a and b two scalars
        """
        return abs(a-b) / (abs(a) + abs(b))

    def _energy_grad_weight_check(self, dh=10e-10):
        """
        Verify that our weights are close to the true values.
        """
        size = self.bias[0]
        x = np.ones(size)
        # # weights
        # current_weights =

        # # biases

        # biases =


if __name__ == "__main__":
    import doctest
    doctest.testmod()
    # Equilibrium.test_update_weights()
    Equilibrium.test_grad_check_state()
