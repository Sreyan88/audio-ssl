import torch
from torch import nn
from efficientnet_pytorch import EfficientNet

from utils import off_diagonal

class AAAI_BARLOW(nn.Module):

    def __init__(self,args):
        super(AAAI_BARLOW, self).__init__()

        self.args = args
        self.units = args.final_units
        self.model_efficient = EfficientNet.from_name('efficientnet-b0',include_top = False, in_channels = 1,image_size = None)
        self.projector = nn.Sequential(nn.Dropout(0.5),nn.Linear(1280, self.units, bias=False),nn.BatchNorm1d(self.units),nn.ReLU(),nn.Linear(self.units, self.units, bias = False))
        self.bn = nn.BatchNorm1d(self.units, affine=False)

    def forward(self,batch1, batch2):

        z1 = self.model_efficient(batch1)
        z2 = self.model_efficient(batch2)
        
        x1 = z1.flatten(start_dim=1)
        x2 = z2.flatten(start_dim=1) #1280 (already swished)

        x1 = self.projector(x1)
        x2 = self.projector(x2)

        c = self.bn(x1).T @ self.bn(x2)

        # sum the cross-correlation matrix between all gpus
        c.div_(self.args.batch_size)
        torch.distributed.all_reduce(c)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()

        loss = on_diag + self.args.lambd * off_diag

        return loss
        