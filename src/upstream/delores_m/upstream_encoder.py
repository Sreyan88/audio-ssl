import torch
from torch import nn

class DELORES_M(nn.Module):
    """
    Encoder for our IEEE JSTSP Special Issue paper:
    Decorrelating feature spaces for learning general-purpose audio representations
    https://ieeexplore.ieee.org/document/9868132
    """

    def __init__(self, encoder, num_classes, cluster_num, n_mels, d):
        super().__init__()

        self.encoder = encoder
        self.fc_2 = nn.Linear(self.encoder.d,num_classes)

    def forward(self, x):
        x = self.encoder(x)

        (x1, _) = torch.max(x, dim=1)
        x2 = torch.mean(x, dim=1)
        x = x1 + x2
        x = self.fc_2(x)

        return x, (x_1,x_2,x_3)