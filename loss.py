import torch.nn as nn
import torch


class CrossEntropyLossCustom(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y, y_hat):
        c_ent = - y * torch.log(y_hat) - (1 - y) * torch.log(1 - y_hat)
        # reshape() for flattening, mean() for getting mean value
        return c_ent.reshape(-1).mean()
