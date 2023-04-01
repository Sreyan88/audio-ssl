import sys
import torch
import torch.nn.functional as F
import torch.nn as nn
import pytorch_lightning as pl
from typing import Union
#from pl_bolts.metrics import mean, precision_at_k
# from torchmetrics import Precision

from models_delores import deloresm_encoder, slicer_encoder, deloress_encoder
from utils import off_diagonal, concat_all_gather, Project_unfused, Classifier_unfused
import contrastive_loss

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
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum().mul(self.scale_loss)
        off_diag = off_diagonal(c).pow_(2).sum().mul(self.scale_loss)
        if self.lambd:
            loss = self.lambd *on_diag + self.lambd * off_diag
        else:
            loss = on_diag + off_diag    
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
        emb_dim: int = 128,
        num_negatives: int = 65536,
        encoder_momentum: float = 0.999,
        softmax_temperature: float = 0.07,
        learning_rate: float = 0.03,
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
        self.model_type = self.arguments.model_type
        # create the encoders
        # num_classes is the output fc dimension
        if self.model_type in ["deloress","unfused"]:
            self.encoder = self.init_encoders(self.model_type)
            if self.model_type == "deloress":
                self.p4 = Projection(2048,None)

            if self.model_type == 'unfused':
                self.p1 = Project_unfused(2048)
                self.p2 = Project_unfused(1024)
                self.p3 = Project_unfused(512)
                self.softmax = nn.Softmax()
                self.classifier = Classifier_unfused(2048, 585) #1251, 11
                self.kl_divg = nn.KLDivLoss(reduction="batchmean")    

        else:    
            self.encoder_q, self.encoder_k = self.init_encoders(self.model_type)
            if use_mlp:  # hack: brute-force replacement
                dim_mlp = self.encoder_q.fc.weight.shape[1]
                self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
                self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)

            for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
                param_k.data.copy_(param_q.data)  # initialize
                param_k.requires_grad = False  # not update by gradient

            # create the queue
            self.register_buffer("queue", torch.randn(emb_dim, num_negatives))
            self.queue = nn.functional.normalize(self.queue, dim=0)

            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
            self.lamb_values = lamb_values
            if self.model_type == 'deloresm':
                self.p1 = Projection(2048,self.lamb_values[0])
                self.p2 = Projection(1024,self.lamb_values[1])
                self.p3 = Projection(512,self.lamb_values[2])
            
            if self.model_type == 'slicier':
                self.criterion_cluster = contrastive_loss.ClusterLoss(128, 1, None)

    def init_encoders(self, model_type):
        """
        Override to add your own encoders
        """
        if model_type == "deloresm":
            encoder_q = deloresm_encoder(self.hparams.emb_dim, 64, 2048)
            encoder_k = deloresm_encoder(self.hparams.emb_dim, 64, 2048)
            return encoder_q, encoder_k
        elif model_type == "moco":
            encoder_q = deloresm_encoder(self.hparams.emb_dim, 64, 2048)
            encoder_k = deloresm_encoder(self.hparams.emb_dim, 64, 2048)
            return encoder_q, encoder_k    
        elif model_type == "slicer":
            encoder_q = slicer_encoder(self.hparams.emb_dim, 128, 64, 2048)
            encoder_k = slicer_encoder(self.hparams.emb_dim, 128, 64, 2048)
            return encoder_q, encoder_k
        elif model_type in ["deloress","unfused"]:
            encoder = deloress_encoder(self.hparams.emb_dim, 64, 2048)
            return encoder
        
        

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

    def loss_cluster(self, q_cluster, k_cluster):
        return self.criterion_cluster(q_cluster, k_cluster)
    
    def loss_fn(self, x, y):
        x = F.normalize(x, dim=-1, p=2)
        y = F.normalize(y, dim=-1, p=2)
        l = 2 - 2 * (x * y).sum(dim=-1)
        return l.mean()
    
    def forward(self, img_q=None, img_k=None):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """
        if self.model_type == "deloresm" or self.model_type == "moco":
        # compute query features
            q,(q1,q2,q3) = self.encoder_q(img_q)  # queries: NxC
            q = nn.functional.normalize(q, dim=1)
            # compute key features
            with torch.no_grad():  # no gradient to keys
                self._momentum_update_key_encoder()  # update the key encoder
                # shuffle for making use of BN
                if self.trainer.use_ddp or self.trainer.use_ddp2:
                    img_k, idx_unshuffle = self._batch_shuffle_ddp(img_k)
                k,(k1,k2,k3) = self.encoder_k(img_k)  # keys: NxC
                k = nn.functional.normalize(k, dim=1)
                # undo shuffle
                if self.trainer.use_ddp or self.trainer.use_ddp2:
                    k = self._batch_unshuffle_ddp(k, idx_unshuffle)
            # compute logits
            # Einstein sum is more intuitive
            # positive logits: Nx1
            l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
            # negative logits: NxK
            l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
            # logits: Nx(1+K)
            logits = torch.cat([l_pos, l_neg], dim=1)
            # apply temperature
            logits /= self.hparams.softmax_temperature
            # labels: positive key indicators
            labels = torch.zeros(logits.shape[0], dtype=torch.long)
            labels = labels.type_as(logits)

            # dequeue and enqueue
            self._dequeue_and_enqueue(k)

            return logits, labels, q1,q2,q3,k1,k2,k3
        
        elif self.model_type == "slicer":
            q,q_cluster,(q1,q2,q3) = self.encoder_q(img_q)  # queries: NxC
            q = nn.functional.normalize(q, dim=1)
            # compute key features
            with torch.no_grad():  # no gradient to keys
                self._momentum_update_key_encoder()  # update the key encoder
                # shuffle for making use of BN
                if self.trainer.use_ddp or self.trainer.use_ddp2:
                    img_k, idx_unshuffle = self._batch_shuffle_ddp(img_k)

                k,k_cluster,(k1,k2,k3) = self.encoder_k(img_k)  # keys: NxC
                k = nn.functional.normalize(k, dim=1)

                # undo shuffle
                if self.trainer.use_ddp or self.trainer.use_ddp2:
                    k = self._batch_unshuffle_ddp(k, idx_unshuffle)

            # compute logits
            # Einstein sum is more intuitive
            # positive logits: Nx1
            l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
            # negative logits: NxK
            l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

            # logits: Nx(1+K)
            logits = torch.cat([l_pos, l_neg], dim=1)

            # apply temperature
            logits /= self.hparams.softmax_temperature

            # labels: positive key indicators
            labels = torch.zeros(logits.shape[0], dtype=torch.long)
            labels = labels.type_as(logits)

            # dequeue and enqueue
            self._dequeue_and_enqueue(k)

            return logits, labels, q_cluster, q1,q2,q3, k_cluster, k1,k2,k3
        
        elif self.model_type == 'deloress':
            q,(q1,q2,q3) = self.encoder(img_q)
            k,(k1,k2,k3) = self.encoder(img_k)
            return q, k, q1,q2,q3, k1,k2,k3

        elif self.model_type == 'unfused':
            q_raw, (q1,q2,q3) = self.encoder(img_q)
            q_classifer = self.classifier(q_raw)
        return q1,q2,q3,q_classifer    


    def training_step(self, batch, batch_idx):
        # in STL10 we pass in both lab+unl for online ft
        if self.trainer.datamodule.name == 'stl10':
            unlabeled_batch = batch[0]
            batch = unlabeled_batch
        if self.model_type == 'unfused':
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
            log = {'train_loss': loss_complete, 'kl-loss': loss_kl, 'CE-loss': loss_ce, 'mse-loss': loss_mse}
            # log = {'train_loss': loss, 'train_acc1': acc1, 'train_acc5': acc5}
            self.log_dict(log)
            return loss_complete
        else:
            img_1, img_2 = batch
            if self.model_type == "deloresm" or self.model_type == "moco":
                output, target,q1,q2,q3,k1,k2,k3 = self(img_q=img_1, img_k=img_2)
                loss = F.cross_entropy(output.float(), target.long())
                if self.model_type == "deloresm":
                    loss= loss + self.p1(q1,k1)
                    loss= loss + self.p2(q2,k2)
                    loss= loss + self.p3(q3,k3)
                    log = {'train_loss': loss}
                    self.log_dict(log)
                    return loss
                return loss
            elif self.model_type == "slicer":
                output, target, q_cluster,q1,q2,q3, k_cluster,k1,k2,k3 = self(img_q=img_1, img_k=img_2)
                output_1, target_1, q_cluster_1,q1,q2,q3, k_cluster_1,k1,k2,k3 = self(img_q=img_2, img_k=img_1)
                loss_0 = F.cross_entropy(output.float(), target.long())
                loss_1 = F.cross_entropy(output_1.float(), target_1.long())
                loss_instance = (loss_0+loss_1)/2
                #print('Main loss = {}'.format(loss_instance))
                loss_cluster = self.loss_cluster(q_cluster, q_cluster_1)
                #print('cluster loss = {}'.format(loss_cluster))
                loss = loss_cluster + loss_instance
                log = {'train_loss': loss, 'instance_loss': loss_instance, 'cluster_loss': loss_cluster}
                self.log_dict(log)
                return loss

            elif self.model_type == "deloress":
                q, k, q1,q2,q3, k1,k2,k3 = self(img_q=img_1, img_k=img_2)
                loss = self.p4(q,k)
                log = {'train_loss': loss}
                self.log_dict(log)
                return loss
            


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
