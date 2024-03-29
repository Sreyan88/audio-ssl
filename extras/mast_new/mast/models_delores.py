import re
import logging
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F

from utils import off_diagonal


class NetworkCommonMixIn():
    """Common mixin for network definition."""

    def load_weight(self, weight_file, device):
        """Utility to load a weight file to a device."""

        state_dict = torch.load(weight_file, map_location=device)
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        # Remove unneeded prefixes from the keys of parameters.
        weights = {}
        for k in state_dict:
            m = re.search(r'(^fc\.|\.fc\.|^features\.|\.features\.)', k)
            if m is None: continue
            new_k = k[m.start():]
            new_k = new_k[1:] if new_k[0] == '.' else new_k
            weights[new_k] = state_dict[k]
        # Load weights and set model to eval().
        self.load_state_dict(weights)
        self.eval()
        logging.info(f'Using audio embbeding network pretrained weight: {Path(weight_file).name}')
        return self

    def set_trainable(self, trainable=False):
        for p in self.parameters():
            p.requires_grad = trainable



class AudioNTT2020Task6(nn.Module, NetworkCommonMixIn):
    """DCASE2020 Task6 NTT Solution Audio Embedding Network."""

    def __init__(self, num_classes, cluster_num, n_mels, d):
        super().__init__()
        self.features_1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2))

        self.features_2 = nn.Sequential(nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2))

        self.features_3 = nn.Sequential(nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2))
            
        self.fc = nn.Sequential(
            nn.Linear(64 * (n_mels // (2**3)), d),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(d, d),
            nn.ReLU(),
        )

        self.d = d

        self.cluster_num = cluster_num

        self.fc_2 = nn.Linear(d,num_classes)

        self.cluster_projector = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, self.cluster_num),
            nn.Softmax(dim=1)
        )


    def forward(self, x):
        x = self.features_1(x)

        x_1 = x.permute(0, 3, 2, 1)
        B, T, D, C = x_1.shape
        x_1 = x_1.reshape((B, T, C*D))
        x_1 = torch.mean(x_1, dim=1)

        x = self.features_2(x)

        x_2 = x.permute(0, 3, 2, 1)
        B, T, D, C = x_2.shape
        x_2 = x_2.reshape((B, T, C*D))
        x_2 = torch.mean(x_2, dim=1)

        x = self.features_3(x) # (batch, ch, mel, time)

        x_3 = x.permute(0, 3, 2, 1)
        B, T, D, C = x_3.shape
        x_3 = x_3.reshape((B, T, C*D))
        x_3 = torch.mean(x_3, dim=1)

        x = x.permute(0, 3, 2, 1) # (batch, time, mel, ch)
        B, T, D, C = x.shape
        x = x.reshape((B, T, C*D))
        
        x = self.fc(x)

        (x1, _) = torch.max(x, dim=1)
        x2 = torch.mean(x, dim=1)
        x = x1 + x2
        x_cluster = self.cluster_projector(x)
        x = self.fc_2(x)

        return x, x_cluster, (x_1,x_2,x_3)

    def forward_cluster(self, x):
        x = self.features_1(x)

        x_1 = x.permute(0, 3, 2, 1)
        B, T, D, C = x_1.shape
        x_1 = x_1.reshape((B, T, C*D))
        x_1 = torch.mean(x_1, dim=1)

        x = self.features_2(x)

        x_2 = x.permute(0, 3, 2, 1)
        B, T, D, C = x_2.shape
        x_2 = x_2.reshape((B, T, C*D))
        x_2 = torch.mean(x_2, dim=1)

        x = self.features_3(x) # (batch, ch, mel, time)

        x_3 = x.permute(0, 3, 2, 1)
        B, T, D, C = x_3.shape
        x_3 = x_3.reshape((B, T, C*D))
        x_3 = torch.mean(x_3, dim=1)

        x = x.permute(0, 3, 2, 1) # (batch, time, mel, ch)
        B, T, D, C = x.shape
        x = x.reshape((B, T, C*D))

        x = self.fc(x)

        (x1, _) = torch.max(x, dim=1)
        x2 = torch.mean(x, dim=1)
        x = x1 + x2
        x_cluster = self.cluster_projector(x)
        c = torch.argmax(x_cluster, dim=1)
        return c



class AudioNTT2020(AudioNTT2020Task6):
    """BYOL-A General Purpose Representation Network.
    This is an extension of the DCASE 2020 Task 6 NTT Solution Audio Embedding Network.
    """

    def __init__(self, args, n_mels=64, d=512):
        super().__init__(n_mels=n_mels, d=d)
        self.args = args
        self.units = args.final_units
        self.projector = nn.Sequential(nn.Dropout(0.5),nn.Linear(2048, self.units, bias=False),nn.BatchNorm1d(self.units),nn.ReLU(),nn.Linear(self.units, self.units, bias = False))
        self.bn = nn.BatchNorm1d(self.units, affine=False)

    def forward(self, batch1, batch2):

        z1 = super().forward(batch1)
        z2 = super().forward(batch2)

        (z1_1, _) = torch.max(z1, dim=1)
        z1_2 = torch.mean(z1, dim=1)
        z1 = z1_1 + z1_2

        (z2_1, _) = torch.max(z2, dim=1)
        z2_2 = torch.mean(z2, dim=1)
        z2 = z2_1 + z2_2

        x1 = self.projector(z1)
        x2 = self.projector(z2)

        c = self.bn(x1).T @ self.bn(x2)

        # sum the cross-correlation matrix between all gpus
        c.div_(self.args.batch_size)
        torch.distributed.all_reduce(c)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()

        loss = on_diag + self.args.lambd * off_diag

        return loss
