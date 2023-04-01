import torch
from torch import nn

class DELORES_S(nn.Module):
    """
    Encoder for our AAAI 2023 workshop paper:
    Delores: Decorrelating latent spaces for low-resource audio representation learning
    https://arxiv.org/pdf/2203.13628.pdf
    """

    def __init__(self, encoder, num_classes, cluster_num, n_mels, d):
        super().__init__()

        self.encoder = encoder

    def forward(self, x):
        x = self.encoder(x)

        (x1, _) = torch.max(x, dim=1)
        x2 = torch.mean(x, dim=1)
        x = x1 + x2

        return x, (x_1,x_2,x_3)