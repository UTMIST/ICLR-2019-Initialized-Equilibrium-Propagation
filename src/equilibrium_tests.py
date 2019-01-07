"""
Tests for EquilibriumNet.
"""

from equilibrium import EquilibriumNet, rho
import torch


def test_energy():
    x = torch.tensor([[1, 2, 3], [4, 5, 6]])
    w1 = torch.tensor([[3, 9, 5], [1, 2, 7], [3, 3, 2]])
    w2 = torch.tensor([[3, 8, 8], [9, 5, 1]])
    w3 = torch.tensor([[9, 2], [2, 0]])
    b = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8])  # for x, l2, l3

    l2 = torch.tensor([9, 4, 6])
    l3 = torch.tensor([5, 3])
    l4 = torch.tensor([4, 5])

    squared_norm = (torch.sum(l2) ** 2 +
                    torch.sum(l3) ** 2 + torch.sum(l4) ** 2) / 2

    first = torch.cat((l2, l3), 0)
    second = torch.cat((first, l4), 0)
    bias_sum = torch.sum(b * rho(second))

    prod1 = torch.dot(torch.mv(w2.T, l3), l2)
    prod2 = torch.dot(torch.mv(w3.T, l4), l3)
    tensor_product = sum(prod1, prod2)

    input_sums = -torch.mv(x, torch.mv(w1.T, l2))  # Apply rho to l2?

    energy = input_sums.add(squared_norm - bias_sum - tensor_product)

    assert True  # compare with class
