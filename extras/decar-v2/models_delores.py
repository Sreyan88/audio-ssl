import re
import sys
import logging
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from utils import MultiPrototypes


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

    def __init__(self, n_mels, d):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * (n_mels // (2**3)), d),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(d, d),
            nn.ReLU(),
        )
        self.d = d

    def forward(self, x):
        x = self.features(x)       # (batch, ch, mel, time)       
        x = x.permute(0, 3, 2, 1) # (batch, time, mel, ch)
        B, T, D, C = x.shape
        x = x.reshape((B, T, C*D)) # (batch, time, mel*ch)
        x = self.fc(x)
        return x


class AudioNTT2020(AudioNTT2020Task6):
    """BYOL-A General Purpose Representation Network.
    This is an extension of the DCASE 2020 Task 6 NTT Solution Audio Embedding Network.
    """

    def __init__(self, args, out_dim, n_mels=64, d=512, nmb_prototypes=3000):
        super().__init__(n_mels=n_mels, d=d)
        self.args = args

        self.projection_head = nn.Sequential(
                nn.Linear(d, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(inplace=True),
                nn.Linear(2048, out_dim),
            )
        #self.prototypes = nn.Linear(out_dim, nmb_prototypes[0], bias=False)
        if isinstance(args.nmb_prototypes, list):
            self.prototypes = MultiPrototypes(out_dim, [1024])
        elif args.nmb_prototypes > 0:
            self.prototypes = nn.Linear(out_dim, 1024, bias=False)

    def forward(self, batch):

        # for first augmentation
        z = super().forward(batch[0])

        (z1, _) = torch.max(z, dim=1)
        z2 = torch.mean(z, dim=1)
        z = z1 + z2

        #for second augmentation
        z_new = super().forward(batch[1])

        (z1, _) = torch.max(z_new, dim=1)
        z2 = torch.mean(z_new, dim=1)
        z_new = z1 + z2

        x = self.projection_head(z)
        x_new = self.projection_head(z_new)

        if len(self.args.crops_for_assign) == 1:
            x = x
        return x, self.prototypes(x_new)
