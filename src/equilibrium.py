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
    >>> EquilibriumNet.rho(torch.tensor([1, 2, -1, -2, 0.5, -0.5]))
    tensor([1.0000, 1.0000, 0.0000, 0.0000, 0.5000, 0.0000])
    """
    return torch.clamp(v, 0, 1)

def rhoprime(v):
    """
    The gradient of the activation function to be used (a hard sigmoid),
    here just a Boolean function denoting whether the value is in the
    clamp range
    >>> EquilibriumNet.rhoprime(torch.tensor([0.9, 2, -1, -2, 0.5, -0.5]))
    tensor([1., 0., 0., 0., 1., 0.])
    """
    return (torch.gt(v, 0) & torch.lt(v, 1)).type(torch.Tensor)

class EquilibriumNet:
    """
    A fully connected feed-forward equilibrium propagation network
    """

    # The shape of the equilibrium propagation network as a fully connected
    # network
    shape : List[int]

    @staticmethod
    def get_default_device():
        """
        Get the default device for the equilibrium network's components
        """
        return None

    def __init__(self,
        input_size : int, layer_sizes : List[int], output_size : int, **kwargs):
        """
        Initialize an equilibrium propagation network with given input size,
        output size and fully connected layer sizes on a given device

        >>> new_net = EquilibriumNet(28*28, [500, 500], 10)
        >>> new_net.shape
        [784, 500, 500, 10]
        >>> len(new_net.state_particles), len(new_net.weights), len(new_net.biases)
        (3, 3, 3)
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
        partial_sums = [0]
        partial_sums.extend(itertools.accumulate(self.shape[1:], operator.add))

        # Initialize the state particles for equilibrium propagation
        self.state_particles = torch.zeros(partial_sums[-1])
        # State particles for the neurons in each individual layer,
        # implemented as views of the memory in state_particles
        self.layer_state_particles = [
            self.state_particle[b:e] for (b, e) in
                zip(partial_sums[:-1], partial_sums[1:])
        ]

        # Initialize the weights
        self.weights = [
            torch.randn(D_in, D_out, device=self.device) for (D_in, D_out) in
                zip(self.shape[:-1], self.shape[1:])
        ]

        # Initialize the bias
        self.biases = torch.randn(partial_sums[-1])
        # Bias for the neurons in each individual layer, implemented as views of
        # the memory in biases
        self.layer_biases = [
            self.biases[b:e] for (b, e) in
                zip(partial_sums[:-1], partial_sums[1:])
        ]

    def energy(self, x):
        """
        The "potential energy" of the equilibrium propagation network for in its
        current state for each input x, where x is a tensor of shape
        (batch_size, input_size)
        """
        assert len(x.shape) == 2 and x.shape[0] == self.shape[0]

        # Squared norm of the state
        # LaTeX: \frac{1}{2}\sum_{i \in \mathcal{S}}s_i^2
        squared_norm = torch.sum(self.state_particles ** 2)


        # Product of bias and state activation
        # LaTeX: \sum_{i \in \mathcal{S}}b_i\rho(s_i)
        bias_sum = torch.sum(
            self.biases * rho(self.state_particles))

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
            torch.mv(W, rho(s_out)) for W, s_out in
                zip(self.weights[1:].T, self.layer_state_particles[1:])
        ]

        # Dot product of said matrix products and the activations of the vectors
        # connected to j, summed over all layers
        tensor_product = sum(
            [torch.dot(pr, rho(s_in)) for pr, s_in in
                zip(next_weights, self.layer_state_particles[:-1])]
        )

        # Tensor product of weight matrix, activation of non-state neurons j and
        # activation of input neurons i connected to j for each input value in x
        #
        # Due to the structure of our network, only neurons in the layer after
        # the first are connected to the input neurons, and hence we need only
        # consider these
        input_sums = -torch.mv(
            x, torch.mv(self.weights[0].T, self.layer_state_particles[0])
        )

        # Now, we compute the energy for each element of x
        return input_sums.add(squared_norm - bias_sum - next_weights)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
    print("Doctests complete")
