"""
Defines an equilibrium propagation network model
"""
import torch

from typing import List

class EquilibriumNet:
    """
    A fully connected equilibrium propagation network
    """

    # The shape of the equilibrium propagation network as a fully connected
    # network
    shape : List[int]

    @staticmethod
    def get_default_device():
        """
        Get the default device for the equilibrium network's components
        """
        return "cpu"

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
            self.device = torch.device(self.get_default_device())

        # Get the shape array
        self.shape = [input_size]
        self.shape.extend(layer_sizes)
        self.shape.append(output_size)

        # Initialize the state particles for equilibrium propagation
        self.state_particles = [
            torch.zeros(D) for D in self.shape[1:]
        ]

        # Initialize the weights
        self.weights = [
            torch.randn(D_in, D_out, device=self.device) for (D_in, D_out) in
                zip(self.shape[:-1], self.shape[1:])
        ]

        # Initialize the bias
        self.biases = [
            torch.randn(D, device=self.device) for D in self.shape[:-1]
        ]


if __name__ == "__main__":
    import doctest
    doctest.testmod()
    print("Doctests complete")
