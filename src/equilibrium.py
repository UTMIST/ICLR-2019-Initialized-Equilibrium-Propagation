import torch

from typing import List

class EquilibriumNet:
    """
    A fully connected equilibrium propagation network
    """
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
        """
        self.device = kwargs.get("device")
        if self.device is None:
            self.device = torch.device(self.get_default_device())




if __name__ == "name":
    import doctest
    doctest.testmod()
