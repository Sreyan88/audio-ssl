import sys
import torch
import torch.nn.functional as F
import torch.nn as nn
import pytorch_lightning as pl
from typing import Union
#from pl_bolts.metrics import mean, precision_at_k
# from torchmetrics import Precision


from models_delores import AudioNTT2020Task6
from utils import off_diagonal, concat_all_gather

from functools import partial
from torch import nn

'''
class ConvNormAct(nn.Sequential):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: int,
        norm: nn.Module = nn.BatchNorm2d,
        act: nn.Module = nn.ReLU,
        **kwargs
    ):

        super().__init__(
            nn.Conv2d(
                in_features,
                out_features,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            norm(out_features),
            act(),
        )


class ConvNormAct(nn.Module):
    def __init__(self, in_features: int, out_features: int, kernel_size: int):
        super().__init__()
        self.conv_layer = nn.Conv2d(in_features, out_features, kernel_size=kernel_size, padding=kernel_size // 2)
        self.norm = nn.BatchNorm2d(out_features)
        self.act = nn.ReLU()
    def forward(self, x):
        x = self.act(self.norm(self.conv_layer(x)))
        return(x)

        
Conv1X1BnReLU = partial(ConvNormAct, kernel_size=1)
Conv3X3BnReLU = partial(ConvNormAct, kernel_size=3)
'''

class Project(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        sizes = [in_dim, 585, 585, 585] #1251
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)
    def forward(self, x):
        return self.projector(x)

class Classifier(nn.Module):
    def __init__(self, in_dim, num_cluster):
        super().__init__()
        self.linear = nn.Linear(in_dim, num_cluster)
        #self.softmax = nn.Softmax()

    def forward(self, x):
        return(self.linear(x))    

class Projection(nn.Module):
    def __init__(self, in_dim,lambd=5e-5,scale_loss=1/32):
        super().__init__()
        # projector
        sizes = [in_dim, 2048, 2048,2048]
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)
        self.lambd=lambd
        self.scale_loss=scale_loss

        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)
        

    def forward(self, y1, y2):
        z1 = self.projector(y1)
        z2 = self.projector(y2)
        batch_size = z1.shape[0]

        # empirical cross-correlation matrix
        c = self.bn(z1).T @ self.bn(z2)

        # sum the cross-correlation matrix between all gpus
        c.div_(batch_size)
#         torch.distributed.all_reduce(c)

        # use --scale-loss to multiply the loss by a constant factor
        # see the Issues section of the readme
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum().mul(self.scale_loss)
        off_diag = off_diagonal(c).pow_(2).sum().mul(self.scale_loss)
        loss = self.lambd *on_diag + self.lambd * off_diag
#         print(on_diag)
#         print(off_diag)
        return loss

# precision = Precision() 



class Moco_v2(pl.LightningModule):
    """
    PyTorch Lightning implementation of `Moco <https://arxiv.org/abs/2003.04297>`_
    Paper authors: Xinlei Chen, Haoqi Fan, Ross Girshick, Kaiming He.
    Code adapted from `facebookresearch/moco <https://github.com/facebookresearch/moco>`_ to Lightning by:
        - `William Falcon <https://github.com/williamFalcon>`_
    Example::
        from pl_bolts.models.self_supervised import Moco_v2
        model = Moco_v2()
        trainer = Trainer()
        trainer.fit(model)
    CLI command::
        # cifar10
        python moco2_module.py --gpus 1
        # imagenet
        python moco2_module.py
            --gpus 8
            --dataset imagenet2012
            --data_dir /path/to/imagenet/
            --meta_dir /path/to/folder/with/meta.bin/
            --batch_size 32
    """

    def __init__(
        self,
        arguments,
        base_encoder: Union[str, torch.nn.Module] = 'resnet18',
        emb_dim: int = 128,
        num_negatives: int = 65536,
        encoder_momentum: float = 0.999,
        softmax_temperature: float = 0.07,
        learning_rate: float = 0.007,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        data_dir: str = './',
        batch_size: int = 256,
        use_mlp: bool = False,
        num_workers: int = 8,
        lamb_values = [0.25,0.25,0.25,0.25],
        *args,
        **kwargs
    ):
        """
        Args:
            base_encoder: torchvision model name or torch.nn.Module
            emb_dim: feature dimension (default: 128)
            num_negatives: queue size; number of negative keys (default: 65536)
            encoder_momentum: moco momentum of updating key encoder (default: 0.999)
            softmax_temperature: softmax temperature (default: 0.07)
            learning_rate: the learning rate
            momentum: optimizer momentum
            weight_decay: optimizer weight decay
            datamodule: the DataModule (train, val, test dataloaders)
            data_dir: the directory to store data
            batch_size: batch size
            use_mlp: add an mlp to the encoders
            num_workers: workers for the loaders
        """

        super().__init__()
        self.save_hyperparameters()

        self.arguments = arguments

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = self.init_encoders(base_encoder)

        # create the queue
        #self.register_buffer("queue", torch.randn(emb_dim, num_negatives))
        #self.queue = nn.functional.normalize(self.queue, dim=0)

        #self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.lamb_values = lamb_values
        self.p1 = Project(2048)
        self.p2 = Project(1024)
        self.p3 = Project(512)
        self.softmax = nn.Softmax()
        self.classifier = Classifier(2048, 585) #1251
        self.kl_divg = nn.KLDivLoss(reduction="batchmean")

    def init_encoders(self, base_encoder):
        """
        Override to add your own encoders
        """
        encoder_q = AudioNTT2020Task6(self.hparams.emb_dim, 64, 2048)
        #encoder_k = AudioNTT2020Task6(self.hparams.emb_dim, 64, 2048)

        return encoder_q

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            em = self.hparams.encoder_momentum
            param_k.data = param_k.data * em + param_q.data * (1. - em)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        if self.trainer.use_ddp or self.trainer.use_ddp2:
            keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.hparams.num_negatives % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.hparams.num_negatives  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):  # pragma: no cover
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):  # pragma: no cover
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, img_q):
        """
        Input:
            im_q: a batch of query images
        Output:
            logits, targets
        """

        # compute query features
        q_raw, (q1,q2,q3) = self.encoder_q(img_q)  # queries: NxC
        #q = nn.functional.normalize(q, dim=1)
        q_classifer = self.classifier(q_raw)

        return q1,q2,q3,q_classifer

    def loss_fn(self, x, y):
        x = F.normalize(x, dim=-1, p=2)
        y = F.normalize(y, dim=-1, p=2)
        l = 2 - 2 * (x * y).sum(dim=-1)
        #print(l)
        #print(l.shape)
        return l.mean()


    def training_step(self, batch, batch_idx):
        # in STL10 we pass in both lab+unl for online ft
        if self.trainer.datamodule.name == 'stl10':
            # labeled_batch = batch[1]
            unlabeled_batch = batch[0]
            batch = unlabeled_batch

        (img_1, _), label = batch
        #print('image-1 ', img_1.shape)
        #print('image-2 ', img_2.shape)
        #print('label ', label.shape)

        q1,q2,q3,q_classifier = self(img_q=img_1)
        #print('output ', output.shape)
        #print('target ', target.shape)
        q1_tag = self.p1(q1) #new projector for sssd
        q2_tag = self.p2(q2) #new projector for sssd
        q3_tag = self.p3(q3) #new projector for sssd
        #Cross entroy loss defined
        loss_ce1 = F.cross_entropy(q1_tag, label.long())
        loss_ce2 = F.cross_entropy(q2_tag, label.long())
        loss_ce3 = F.cross_entropy(q3_tag, label.long())
        loss_ce = 0.7*loss_ce1 + 0.7*loss_ce2 + 0.7*loss_ce3 + F.cross_entropy(q_classifier, label.long())
        #KL-divergence
        q1_log_soft = F.log_softmax(q1_tag, dim=1)
        q2_log_soft = F.log_softmax(q2_tag, dim=1)
        q3_log_soft = F.log_softmax(q3_tag, dim=1)
        targets = F.softmax(q_classifier, dim=1)
        loss_kl = 0.3*self.kl_divg(q1_log_soft, targets)+0.3*self.kl_divg(q2_log_soft, targets)+0.3*self.kl_divg(q3_log_soft, targets)
        #MSE_loss
        loss_mse = 0.003*(self.loss_fn(q1_tag, q_classifier) + self.loss_fn(q2_tag, q_classifier) + self.loss_fn(q3_tag, q_classifier))
        #final loss
        loss_complete = loss_mse + loss_kl + loss_ce

        print('Main loss = {}, KL-loss = {}, CE-loss = {}, mse-loss = {}'.format(loss_complete, loss_kl, loss_ce, loss_mse))
        
        # print(q1.shape)
        # print(q2.shape)
        # print(q3.shape)

        #loss= loss + self.p1(q1,k1)
        #loss= loss + self.p2(q2,k2)
        #loss= loss + self.p3(q3,k3)
        #loss+= self.p4(q4,k4)

        # print(output)
        # print(target)

        # acc1 = precision_at_k(output, target, top_k=(1, 5))

        # acc1 = precision(output, target)

        log = {'train_loss': loss_complete, 'kl-loss': loss_kl, 'CE-loss': loss_ce, 'mse-loss': loss_mse}
        # log = {'train_loss': loss, 'train_acc1': acc1, 'train_acc5': acc5}
        self.log_dict(log)
        return loss_complete

    def validation_step(self, batch, batch_idx):
        # in STL10 we pass in both lab+unl for online ft
        if self.trainer.datamodule.name == 'stl10':
            # labeled_batch = batch[1]
            unlabeled_batch = batch[0]
            batch = unlabeled_batch

        (img_1, img_2), labels = batch

        output, target,q1,q2,q3,q4,k1,k2,k3,k4 = self(img_q=img_1, img_k=img_2)
        loss = F.cross_entropy(output.float(), target.long())
#         print('Main loss = {}'.format(loss))

        loss+= self.p1(q1,k1)
        loss+= self.p2(q2,k2)
        loss+= self.p3(q3,k3)
        loss+= self.p4(q4,k4)

        acc1, acc5 = precision_at_k(output, target, top_k=(1, 5))

        results = {'val_loss': loss, 'val_acc1': acc1, 'val_acc5': acc5}
        return results

    def validation_epoch_end(self, outputs):
        val_loss = mean(outputs, 'val_loss')
        val_acc1 = mean(outputs, 'val_acc1')
        val_acc5 = mean(outputs, 'val_acc5')

        log = {'val_loss': val_loss, 'val_acc1': val_acc1, 'val_acc5': val_acc5}
        self.log_dict(log)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            self.hparams.learning_rate,
            momentum=self.hparams.momentum,
            weight_decay=self.hparams.weight_decay
        )
        return optimizer
