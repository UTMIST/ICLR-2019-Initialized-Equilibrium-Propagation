"""
Defines an equilibrium propagation network model
"""
import torch
import itertools
import operator

from typing import List

def rho(v):
    """
    The activation function to be used, here a hard sigmoid
    >>> rho(torch.tensor([1, 2, -1, -2, 0.5, -0.5]))
    tensor([1.0000, 1.0000, 0.0000, 0.0000, 0.5000, 0.0000])
    """
    return torch.clamp(v, 0, 1)

def rhoprime(v):
    """
    The gradient of the activation function to be used (a hard sigmoid),
    here just a Boolean function denoting whether the value is in the
    clamp range
    >>> rhoprime(torch.tensor([0.9, 2, -1, -2, 0.5, -0.5]))
    tensor([1., 0., 0., 0., 1., 0.])
    """
    return ((1 - torch.lt(v, 0)) & (1 - torch.gt(v, 1))).type(torch.Tensor)

class EquilibriumNet:
    """
    A fully connected feed-forward equilibrium propagation network
    """

    # The shape of the equilibrium propagation network as a fully connected
    # network
    # shape: List[int]
    # partial_sums: List[int]

    # The size of minibatches to be fed into the equilibrium propagation network
    # minibatch_size : int

    @staticmethod
    def get_default_device():
        """
        Get the default device for the equilibrium network's components
        """
        return "cpu"

    def set_batch_size(self, minibatch_size : int, **kwargs):
        """
        Reinitialize the state with the given minibatch size
        """
        self.minibatch_size = minibatch_size

        self.minibatch_size = minibatch_size

        # Initialize the state particles for equilibrium propagation
        self.state_particles = kwargs.get("initial_state")
        if self.state_particles is None:
            self.state_particles =\
                torch.zeros(self.partial_sums[-1], minibatch_size)
        elif len(self.state_particles.shape) == 1:
            self.state_particles = torch.t(
                self.state_particles.repeat(minibatch_size, 1))
            assert self.state_particles.shape[0] == self.partial_sums[-1],\
                "Got shape {}, expected length {}".format(
                    self.state_particles.shape, self.partial_sums[-1]
                )
        else:
            assert tuple(self.state_particles.shape) == (
                self.partial_sums[-1], minibatch_size
            ), "Bad shape: {}".format(tuple(self.state_particles.shape))

        # State particles for the neurons in each individual layer,
        # implemented as views of the memory in state_particles
        self.layer_state_particles = [
            self.state_particles[b:e] for (b, e) in
                zip(self.partial_sums[:-1], self.partial_sums[1:])
        ]

    def __init__(self,
        input_size : int, layer_sizes : List[int], output_size : int,
        minibatch_size : int, **kwargs):
        """
        Initialize an equilibrium propagation network with given input size,
        output size and fully connected layer sizes on a given device

        >>> new_net = EquilibriumNet(28*28, [500, 500], 10, 32)
        >>> new_net.shape
        [784, 500, 500, 10]
        >>> len(new_net.state_particles), len(new_net.biases)
        (1010, 1010)
        >>> len(new_net.weights), len(new_net.layer_biases)
        (3, 3)
        >>> len(new_net.input_weights)
        1784
        """
        self.device = kwargs.get("device")
        if self.device is None:
            self.device = self.get_default_device()

        # Get the shape array
        self.shape = [input_size]
        self.shape.extend(layer_sizes)
        self.shape.append(output_size)

        # Get the number of particles and bias parameters before each layer, and
        # one past-the-end
        self.partial_sums = [0]
        self.partial_sums.extend(
            itertools.accumulate(self.shape[1:], operator.add))

        self.set_batch_size(minibatch_size,
            initial_state = kwargs.get("initial_state"))

        # Initialize the weights
        self.weights = kwargs.get("weights")
        if self.weights is None:
            self.weights = [
                torch.randn(D_out, D_in, device=self.device) for (D_in, D_out) in
                    zip(self.shape[:-1], self.shape[1:])
            ]
        else:
            assert len(self.weights) == len(self.shape) - 1
            for weights, D_in, D_out in\
                zip(self.weights, self.shape[:-1], self.shape[1:]):
                assert weights.shape == (D_out, D_in),\
                    "Got weight shape {}, expected {}".format(
                        weights.shape, (D_out, D_in))

        # Get a vector of input neurons weights for each state neuron
        self.input_weights = list(
            itertools.chain.from_iterable(
                [[w[p] for p in range(len(w))] for w in self.weights]
            )
        )

        # Initialize the bias
        self.biases = kwargs.get("biases")
        if self.biases is None:
            self.biases = torch.randn(self.partial_sums[-1])
        else:
            assert len(self.biases) == self.partial_sums[-1]

        # Bias for the neurons in each individual layer, implemented as views of
        # the memory in biases
        self.layer_biases = [
            self.biases[b:e] for (b, e) in
                zip(self.partial_sums[:-1], self.partial_sums[1:])
        ]

    def energy(self, x):
        """
        The "potential energy" of the equilibrium propagation network for in its
        current state for each input in x, where x is a tensor of shape
        (minibatch_size, input_size)
        """
        # 2nd argument was self.input_size. -Matt
        assert x.shape == (self.minibatch_size, self.shape[0])

        # Squared norm of the state
        # LaTeX: \frac{1}{2}\sum_{i \in \mathcal{S}}s_i^2
        squared_norm = torch.sum(self.state_particles ** 2, dim=0) / 2


        # Product of bias and state activation
        # LaTeX: \sum_{i \in \mathcal{S}}b_i\rho(s_i)
        bias_sum = torch.matmul(self.biases, rho(self.state_particles))

        # Tensor product of weight matrix, activation of non-state neurons j and
        # activation of non-state neurons i connected to j
        #
        # Due to the structure of our network (feed-forward), neurons in layers
        # after the first, potentially including the output (last) layer, are
        # connected to state neurons in the previous layer, giving the form of
        # our calculation


        # Matrix product of the weight matrix for a layer and the activation of
        # neurons in that layer.
        next_weights = [
            torch.mm(torch.t(W), rho(s_out)) for W, s_out in
                zip(self.weights[1:], self.layer_state_particles[1:])
        ]

        # Dot product of said matrix products and the activations of the vectors
        # connected to j, summed over all layers
        tensor_product = sum(
            [torch.matmul(torch.t(pr), rho(s_in)).diag() for pr, s_in in
                zip(next_weights, self.layer_state_particles[:-1])]
        )

        # Tensor product of weight matrix, activation of non-state neurons j and
        # activation of input neurons i connected to j for each input value in x
        #
        # Due to the structure of our network, only neurons in the layer after
        # the first are connected to the input neurons, and hence we need only
        # consider these
        input_sums = -torch.mm(
            x, torch.mm(torch.t(self.weights[0]), rho(self.layer_state_particles[0]))
        ).diag()

        # Now, we compute the energy for each element of x
        return input_sums + squared_norm - bias_sum - tensor_product

    def energy_grad_state(self, x):
        """
        Gradient of energy with respect to each component of the current state
        for each component of the minibatch x
        """
        assert x.shape == (self.minibatch_size, self.shape[0])

        # Get the derivative of the activation for the state for each batch
        act_prime_states = rhoprime(self.state_particles)

        # print("weight: {}".format([w.shape for w in self.weights]))
        # print("layer: {}".format([l.shape for l in self.layer_state_particles]))
        #
        weight_product = [
            torch.matmul(self.weights[0], torch.t(x))
        ]
        weight_product += [torch.matmul(weights, layer) for weights, layer in
               zip(self.weights[1:], self.layer_state_particles[:-1])]
        #
        # print("weight_product: {}".format([r.shape for r in weight_product]))
        #
        # weight_product = torch.cat(weight_product)
        # input_product = torch.nn.functional.pad(
        #         torch.matmul(self.weights[0], torch.t(x)),
        #         (0, self.partial_sums[-1] - self.shape[0])
        #     )

        # print("weight_product: {}, input: {}".format(weight_product.shape, input_product.shape))

        biases = torch.clone(self.biases)  # TODO: note sure if unsqueeze, repeat in-place?
        return (self.state_particles - act_prime_states * (biases.unsqueeze(dim=1).repeat(1, self.minibatch_size) + torch.cat(weight_product))) / 2

    def energy_grad_weight(self, state, x):
        """
        Gradient of energy with respect to the weights
        """
        assert state.shape == self.state_particles.shape

        activated_state = rho(state)

        bias_grad = torch.mean(activated_state, dim=1)

        # TODO: this code is copied from constructor

        layer_states = [torch.t(x)]
        layer_states += [
            activated_state[b:e] for (b, e) in
            zip(self.partial_sums[:-1], self.partial_sums[1:])
        ]

        weight_grad = [
            torch.mm(next_activated_state, torch.t(prev_activated_state)) / self.minibatch_size
            for (prev_activated_state, next_activated_state) in
            zip(layer_states[:-1], layer_states[1:])
        ]

        return weight_grad, bias_grad

    def negative_phase(self, x, t_minus: int, epsilon: float):
        """
        Negative phase training.
        """
        self.state_particles = torch.zeros(self.partial_sums[-1], self.minibatch_size)
        for _ in range(t_minus):
            grad = self.energy_grad_state(x)
            # update the state
            # TODO: shape of the grad? is it negative or positive grad?
            self.state_particles -= epsilon * grad
        return self.state_particles

    def positive_phase(self, x, y, t_plus: int, epsilon: float, beta: float):
        """
        Positive phase training.
        """
        self.state_particles = torch.zeros(self.partial_sums[-1], self.minibatch_size)
        for _ in range(t_plus):
            grad = self.clamped_energy_grad(x, y, beta)
            # update the state
            # TODO: shape of the grad? is it negative or positive grad?
            self.state_particles -= epsilon * grad
        return self.state_particles

    def clamped_energy_grad(self, x, y, beta: float):
        """
        Returns the gradient of the clamped energy function evaluated at the current state and input
        """
        # get weight of gradient
        grad = self.energy_grad_state(x)
        # state particles for each layer
        clamp = beta * (self.layer_state_particles[-1] - y) # TODO: is it 2?
        # want to get the last part
        clamped_grad = grad + torch.cat([torch.zeros(self.partial_sums[-2], self.minibatch_size), clamp])

        return clamped_grad

    def update_weights(self, beta, etas, s_pos, s_neg, x):
        """
        Updates weights based on weights gradient
        """
        # get weight gradient, separate learning rates for each layer
        weight_grad_pos, bias_grad_pos = self.energy_grad_weight(s_pos, x)
        weight_grad_neg, bias_grad_neg = self.energy_grad_weight(s_neg, x)
        # update the weights
        for i in range(len(self.shape) - 1):
            self.weights[i] -= (etas[i]/beta) * (weight_grad_pos[i] - weight_grad_neg[i])
        # update the biases
        # multiply different sections of vector by different learning rates eta
        bias_update = torch.zeros(self.partial_sums[-1])
        for (b, e) in zip(self.partial_sums[:-1], self.partial_sums[1:]):
            bias_update += torch.cat([torch.zeros(b),
                                    (etas[i]/beta) * (bias_grad_pos[b:e] - bias_grad_neg[b:e]),
                                    torch.zeros(self.partial_sums[-1] - e)])
        self.biases -= bias_update
        # TODO: assert update is correct

if __name__ == "__main__":
    import doctest
    doctest.testmod()
    print("Doctests complete")
