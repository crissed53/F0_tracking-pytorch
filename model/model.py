import torch.nn as nn
import torch


class SalomonF0Tracker(nn.Module):
    """
    Convolutional Neural Network used for multiple f0-tracking, proposed by
    Bittner et. al, "Deep Salience Representations for F0 estimation in
    Polyphonic Music", ISMIR 2017.

    The model consists of 5 convolutional layers, each with batch normalization
    and ReLU activation, except the last layer.
    The last layer uses logistic activation (sigmoid) for each bin.
    """

    def __init__(self):
        super().__init__()
        # Input size: 6 x 360 x 50 (h, f, t)
        self.conv_net = nn.Sequential(
            # Layer 1, 6 -> 128 filters, 5x5 filter size
            nn.Conv2d(6, 128, 5, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            # Layer 2, 128 -> 64 filters, 5x5 filter size
            nn.Conv2d(128, 64, 5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            # Layer 3, 64 -> 64 filters, 3x3 filter size
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            # Layer 4 (Same as layer 3), 64 -> 64 filters, 3x3 filter size
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            # Layer 5, 64 -> 64 filters, 69x3 filter size
            # In paper, filter size of 70x3 is used, but since unbalanced
            # zero padding is to be applied to preserve input/output dimension,
            # filter size of 69x3 is used instead
            nn.Conv2d(64, 8, (69, 3), padding=(34, 1)),
            nn.BatchNorm2d(8),
            nn.ReLU(True),

            # Final downsampling layer, logistic regression on each bin of
            # the latest activation
            nn.Conv2d(8, 1, 1)
        )

        self.init_layers()

    def init_layers(self):
        for layer in self.conv_net:
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor):
        x = self.conv_net(x)
        return torch.sigmoid(x)


if __name__ == '__main__':
    net = SalomonF0Tracker()
    _input = torch.randn(64, 6, 360, 50)
    _output = net(_input)
    print(f'input shape: {_input.shape}')
    print(f'output shape: {_output.shape}')
