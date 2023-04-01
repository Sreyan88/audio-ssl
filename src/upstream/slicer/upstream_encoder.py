import torch
from torch import nn

class SLICER(nn.Module):
    """
    Encoder for our ICASSP 2023 paper:
    SLICER: LEARNING UNIVERSAL AUDIO REPRESENTATIONS USING LOW-RESOURCE SELF-SUPERVISED PRE-TRAINING
    https://arxiv.org/pdf/2211.01519.pdf
    """

    def __init__(self, encoder, num_classes, cluster_num, n_mels, d):
        super().__init__()

        self.encoder_instance = encoder

        self.encoder_cluster = encoder

        self.cluster_num = cluster_num

        self.fc_2 = nn.Linear(d,num_classes)

        self.cluster_projector = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, self.cluster_num),
            nn.Softmax(dim=1)
        )

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


    def forward(self, x):
        x = self.encoder_instance(x)

        (x1, _) = torch.max(x, dim=1)
        x2 = torch.mean(x, dim=1)
        x = x1 + x2
        x_cluster = self.cluster_projector(x)
        x = self.fc_2(x)

        return x, x_cluster, (x_1,x_2,x_3)