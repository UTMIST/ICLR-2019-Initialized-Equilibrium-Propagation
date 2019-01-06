from equilibrium import Equilibrium
import numpy as np

def test_grad_check_state():
    """
    Checking gradient computation wrt states.
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
