import re
import logging
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F

#from utils import off_diagonal


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
            nn.Dropout(p=0.2),
            nn.Linear(d, d),
            nn.ReLU(),
        )

        self.d = d

        #self.fc_2 = nn.Linear(d,num_classes)

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

        #(x1, _) = torch.max(x, dim=1)
        #x2 = torch.mean(x, dim=1)
        #x = x1 + x2

        #x = self.fc_2(x)

        return x,x_1,x_2,x_3

class AudioNTT2020(AudioNTT2020Task6):
    """BYOL-A General Purpose Representation Network.
    This is an extension of the DCASE 2020 Task 6 NTT Solution Audio Embedding Network.
    """

    def __init__(self, args, n_mels=64, d=2048, no_of_classes=1):
        super().__init__(n_mels=n_mels, d=d)
        self.args = args
        if args.layer_prob == 1:
            self.final = nn.Linear(2048,no_of_classes)
        elif args.layer_prob == 2:
            self.final = nn.Linear(1024,no_of_classes)
        elif args.layer_prob == 3:
            self.final = nn.Linear(512,no_of_classes)
        else:
            self.final = nn.Linear(d,no_of_classes)

    def forward(self, x):
        x,x_1,x_2,x_3 = super().forward(x)
        if self.args.layer_prob == 1:
            output = self.final(x_1)
        elif self.args.layer_prob == 2:
            output = self.final(x_2)
        elif self.args.layer_prob == 3:
            output = self.final(x_3)
        else:    
            #(x1, _) = torch.max(x, dim=1)
            x2 = torch.mean(x, dim=1)
            x = x2 #x1 + x2
            assert x.shape[1] == self.d and x.ndim == 2
            output = self.final(x)
        return output

    def forward_intermediate(self,x):
        x,x_1,x_2,x_3 = super().forward(x)
        return [x_1,x_2,x_3]
        
