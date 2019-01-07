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
    b = torch.tensor([1, 2, 3, 4, 5, 6, 7])  # for l2, l3, l4

    l2 = torch.tensor([9, 4, 6])
    l3 = torch.tensor([5, 3])
    l4 = torch.tensor([4, 5])

    squared_norm = (torch.sum(l2) ** 2 +
                    torch.sum(l3) ** 2 + torch.sum(l4) ** 2) / 2

    first = torch.cat((l2, l3), 0)
    second = torch.cat((first, l4), 0)
    bias_sum = torch.sum(b * rho(second))

    w2T = torch.transpose(w2, 0, 1)
    w3T = torch.transpose(w3, 0, 1)
    w1T = torch.transpose(w1, 0, 1)

    prod1 = torch.dot(torch.mv(w2T, rho(l3)), rho(l2))
    prod2 = torch.dot(torch.mv(w3T, rho(l4)), rho(l3))
    tensor_product = prod1 + prod2

    input_sums = -torch.mv(x, torch.mv(w1T, rho(l2)))

    expected_energy = input_sums.add(squared_norm - bias_sum - tensor_product)

    net = EquilibriumNet(3, [3, 2], 2, 2)

    net.biases = torch.tensor([float(x) for x in b])
    net.weights = [w1, w2, w3]
    net.layer_state_particles = [l2, l3, l4]

    actual_energy = net.energy(x)

    assert expected_energy == actual_energy


def test_energy_grad_state():
    """
    Check energy_grad_state is correct using finite differences.
    """
    x = torch.tensor([[1, 2, 3], [4, 5, 6]])
    w1 = torch.tensor([[3, 9, 5], [1, 2, 7], [3, 3, 2]])
    w2 = torch.tensor([[3, 8, 8], [9, 5, 1]])
    w3 = torch.tensor([[9, 2], [2, 0]])
    b = torch.tensor([1, 2, 3, 4, 5, 6, 7])  # for l2, l3, l4

    l2 = torch.tensor([9, 4, 6])
    l3 = torch.tensor([5, 3])
    l4 = torch.tensor([4, 5])

    network = EquilibriumNet(3, [3, 2], 2, 2)
    network.weights = [w1, w2, w3]
    network.biases = b
    network.state_particles = torch.cat([l2, l3, l4])

    # perform gradient checking with finite differences
    dh = 10e-5
    for i in range(3 + 2 + 2):
        # perturb one entry in state
        network.state_particles[i] += dh
        f_plus = network.energy(x)
        network.state_particles[i] -= 2 * dh
        f_minus = network.energy(x)
        network.state_particles[i] += dh

        # grad estimate with finite differences
        grad_check = (f_plus - f_minus) / (2 * dh)

        true_grad = network.energy_grad_state(x)  # TODO: implement

        assert relative_error(grad_check, true_grad) < 10e-6


def test_energy_grad_weight():
    """
    Check energy_grad_weight using finite differences.
    """
    x = torch.tensor([[1, 2, 3], [4, 5, 6]])
    w1 = torch.tensor([[3, 9, 5], [1, 2, 7], [3, 3, 2]])
    w2 = torch.tensor([[3, 8, 8], [9, 5, 1]])
    w3 = torch.tensor([[9, 2], [2, 0]])
    b = torch.tensor([1, 2, 3, 4, 5, 6, 7])  # for l2, l3, l4

    l2 = torch.tensor([9, 4, 6])
    l3 = torch.tensor([5, 3])
    l4 = torch.tensor([4, 5])

    network = EquilibriumNet(3, [3, 2], 2, 2)
    network.weights = [w1, w2, w3]
    network.biases = b
    network.state_particles = torch.cat([l2, l3, l4])

    # perform gradient checking with finite differences
    dh = 10e-5
    for (layer, D_in, D_out) in zip(range(1, len(network.shape)), network.shape[:-1], network.shape[1:]):
        for i in range(D_in):
            for j in range(D_out):
                # perturb one entry in state
                network.weights[layer][i][j] += dh
                f_plus = network.energy(x)
                network.weights[layer][i][j] -= 2 * dh
                f_minus = network.energy(x)
                network.weights[layer][i][j] += dh

                # grad estimate with finite differences
                grad_check = (f_plus - f_minus) / (2 * dh)

                true_grad = network.energy_grad_weight(x)  # TODO: implement

                assert relative_error(grad_check, true_grad) < 10e-6


def relative_error(a, b):
    """
    Compute relative error between a and b.
    """
    return abs(a - b) / (abs(a) + abs(b))


if __name__ == "__main__":
    test_energy()
    test_energy_grad_state()
    test_energy_grad_weight()
