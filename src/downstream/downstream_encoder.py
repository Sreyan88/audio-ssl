import re
import logging
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F

class ENCODER(nn.Module):
    """
    Encoder for our IEEE JSTSP Paper:
    Decorrelating Feature Spaces for Learning General-Purpose Audio Representations
    https://ieeexplore.ieee.org/document/9868132
    """
    
    def __init__(self, config, args, base_encoder, no_of_classes):
        super().__init__()
        self.config = config
        self.encoder = base_encoder(config["downstream"]["input"]["n_mels"], config["downstream"]["base_encoder"]["output_dim"], config["downstream"]["base_encoder"]["return_all_layers"])
        if config["downstream"]["layer"] == 1:
            self.final = nn.Linear(2048,no_of_classes)
        elif config["downstream"]["layer"] == 2:
            self.final = nn.Linear(1024,no_of_classes)
        elif config["downstream"]["layer"] == 3:
            self.final = nn.Linear(512,no_of_classes)
        else:
            self.final = nn.Linear(config["downstream"]["base_encoder"]["output_dim"], no_of_classes)
    
    def forward(self, x):

        if repr(self.encoder) == "AudioNTT2020Task6":
            x, x_1, x_2, x_3 = self.encoder(x)
        else:
            raise NotImplementedError("downstream currently supports just AudioNTT2020Task6 encoder")

        if self.config["downstream"]["layer"] == 1:
            output = self.final(x_1)
        elif self.config["downstream"]["layer"] == 2:
            output = self.final(x_2)
        elif self.config["downstream"]["layer"] == 3:
            output = self.final(x_3)
        else:    
            x = torch.mean(x, dim=1)
            output = self.final(x)
        return output