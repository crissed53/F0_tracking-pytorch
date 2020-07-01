import unittest
from network import SalomonF0Tracker
import torch


class NetworkTestCase(unittest.TestCase):
    def test_dimension(self):
        net = SalomonF0Tracker()
        _input = torch.randn(64, 6, 360, 50)
        _output = net(_input)
        print(f'input shape: {_input.shape}')
        print(f'output shape: {_output.shape}')
        self.assertEqual(_input.shape, _output.shape)


if __name__ == '__main__':
    unittest.main()
