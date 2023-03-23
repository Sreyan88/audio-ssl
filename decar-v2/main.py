import argparse
import os
import pickle
import time
import sys
import numpy as np
import torch.backends.cudnn as cudnn
from sklearn.metrics.cluster import normalized_mutual_info_score
from os.path import join as path_join
import json
import torch
import logging
from torch import nn
import pandas as pd
from apex.parallel.LARC import LARC
import math
import shutil


from utils import extract_log_mel_spectrogram, compute_features, get_upstream_parser, AverageMeter, UnifLabelSampler, Logger, init_memory, cluster_memory
from specaugment import specaug
from datasets import collate_fn_padd, BARLOW
from multi_proc import LARS, cosine_scheduler
from augmentations import MixupBYOLA, RandomResizeCrop, RunningNorm
from models_delores import AudioNTT2020
from dino_loss import DINOLoss

#list_of_files_directory_1 = os.listdir("/speech/srayan/icassp/kaggle_data/audioset_train/train_wav/")
#list_of_files_directory = ["/speech/srayan/icassp/kaggle_data/audioset_train/train_wav/" + item for item in list_of_files_directory_1]

AUDIO_SR = 16000

logging.basicConfig(filename='dcv2_l2_l2.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
logger=logging.getLogger()
logger.setLevel(logging.INFO)

class AugmentationModule:
    """BYOL-A augmentation module example, the same parameter with the paper."""

    def __init__(self, args, size, epoch_samples, log_mixup_exp=True, mixup_ratio=0.4):
        self.train_transform = nn.Sequential(
            MixupBYOLA(ratio=mixup_ratio, log_mixup_exp=log_mixup_exp),
            RandomResizeCrop(virtual_crop_scale=(1.0, 1.5), freq_scale=(0.6, 1.5), time_scale=(0.6, 1.5)),
        )
        self.pre_norm = RunningNorm(epoch_samples=epoch_samples)
        print('Augmentations:', self.train_transform)
        self.norm_status = args.use_norm
    def __call__(self, x):
        if self.norm_status == "byol":
            x = self.pre_norm(x)
        return [self.train_transform(x), self.train_transform(x)]

def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def main(gpu, args):

    args.rank += gpu
    
    torch.manual_seed(31)
    torch.cuda.manual_seed_all(31)
    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = True

    np.random.seed(31)

    torch.distributed.init_process_group(
        backend='nccl', init_method=args.dist_url,
        world_size=args.world_size, rank=args.rank)

    list_of_files_directory = pd.read_csv(args.input)
    list_of_files_directory = list(list_of_files_directory["files"])



    #----------------------------------------------------------------------------------#
    # Defining the model

    model = AudioNTT2020(args, args.feat_dim, n_mels=64, d=2048, nmb_prototypes=args.nmb_prototypes).cuda(gpu)

    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu],find_unused_parameters=True)

    if args.rank == 0:
        logger.info(model)
    logger.info("Building model done.")

    #----------------------------------------------------------------------------------#
    # Defining optimizer, augmentation module, datasets and loaders

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.base_lr,
        momentum=0.9,
        weight_decay=1e-6,
    )

    tfms = AugmentationModule(args, (64, 96), 2 * len(list_of_files_directory))
    train_dataset = BARLOW(args, list_of_files_directory, tfms)
    sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

    per_device_batch_size = args.batch_size // args.world_size

    loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=per_device_batch_size, num_workers=32,
        pin_memory=True, sampler=sampler)


    optimizer = LARC(optimizer=optimizer, trust_coefficient=0.001, clip=False)
    
    logger.info("Building optimizer and dataloader done.")

    #----------------------------------------------------------------------------------#
    # Building schedulers

    warmup_lr_schedule = np.linspace(0, args.base_lr, len(loader) * 10)
    iters = np.arange(len(loader) * (args.epochs - 10))
    cosine_lr_schedule = np.array([args.final_lr + 0.5 * (args.base_lr - args.final_lr) * (1 + \
                         math.cos(math.pi * t / (len(loader) * (args.epochs - 10)))) for t in iters])
    lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))

    logger.info("Building schedulers done.")

    #----------------------------------------------------------------------------------#
    # Building memory bank

    mb_path = os.path.join(args.dump_path, "mb" + str(args.rank) + ".pth")
    if os.path.isfile(mb_path):
        mb_ckp = torch.load(mb_path)
        local_memory_index = mb_ckp["local_memory_index"]
        local_memory_embeddings = mb_ckp["local_memory_embeddings"]
    else:
        local_memory_index, local_memory_embeddings = init_memory(args, loader, model, logger)


    start_epoch = 0

    # #Resume from checkpoint
    # if args.resume:
    #     logger.info("loading checkpoint")
    #     checkpoint = torch.load(args.checkpoint_path)
    #     start_epoch = checkpoint['epoch']
    #     model.load_state_dict(checkpoint['state_dict'])
    #     optimizer.load_state_dict(checkpoint['optimizer'])
    #     logger.info("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))

    best_loss = float("inf")

    if args.rank == 0:
        stats_file = open(args.save_dir  + "/" + 'stats_fsd50k.txt', 'a', buffering=1)
    else:
        stats_file = None

    for epoch in range(start_epoch,args.epochs):

        sampler.set_epoch(epoch)
    
        if args.rank == 0:
            logger.info("Starting To Train")

        scores, local_memory_index, local_memory_embeddings = train(
            args,
            loader,
            model,
            optimizer,
            epoch,
            lr_schedule,
            local_memory_index,
            local_memory_embeddings,
            stats_file,
        )


        if args.rank == 0:
            save_dict = {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            torch.save(
                save_dict,
                os.path.join(args.save_dir, 'checkpoints_deepcluster', 'checkpoint_' + str(epoch + 1) + "_" + '.pth.tar'),
            )
            '''
            if epoch % args.checkpoint_freq == 0 or epoch == args.epochs - 1:
                shutil.copyfile(
                    os.path.join(args.dump_path, "checkpoint.pth.tar"),
                    os.path.join(args.dump_checkpoints, "ckp-" + str(epoch) + ".pth"),
                )
            '''    
        torch.save({"local_memory_embeddings": local_memory_embeddings,
                    "local_memory_index": local_memory_index}, mb_path)

        

def train(args, loader, model, optimizer, epoch, schedule, local_memory_index, local_memory_embeddings, stats_file):

    batch_time = AverageMeter()
    losses = AverageMeter()
    data_time = AverageMeter()
    forward_time = AverageMeter()
    backward_time = AverageMeter()
    print('model training started')
    model.train()
    print('model training completed')
    cross_entropy = nn.CrossEntropyLoss(ignore_index=-100)

    assignments = cluster_memory(args,model, local_memory_index, local_memory_embeddings, len(loader.dataset))
    print('Clustering for epoch {} done.'.format(epoch))
    logger.info('Clustering for epoch {} done.'.format(epoch))

    end = time.time() 
    start_idx = 0
    for i, (idx, inputs) in enumerate(loader):
        data_time.update(time.time() - end)

        it = len(loader) * epoch + i


        # ============ multi-res forward passes ... ============
        emb, output = model(inputs)
        emb = emb.detach()
        bs = inputs[1].size(0)

        # ============ deepcluster-v2 loss ... ============
        loss = 0
        for h in range(len(args.nmb_prototypes)):
            scores = output[h] / 1.0
            targets = assignments[h][idx].repeat(sum(args.nmb_crops)).cuda(non_blocking=True)
            loss += cross_entropy(scores, targets)
        loss /= len(args.nmb_prototypes)

        # ============ backward and optim step ... ============
        optimizer.zero_grad()
        loss.backward()
        # cancel some gradients
        if it < args.freeze_prototypes_niters:
            for name, p in model.named_parameters():
                if "prototypes" in name:
                    p.grad = None
        optimizer.step()

        # ============ update memory banks ... ============
        local_memory_index[start_idx : start_idx + bs] = idx
        for i, crop_idx in enumerate(args.crops_for_assign):
            local_memory_embeddings[i][start_idx : start_idx + bs] = \
                emb[crop_idx * bs : (crop_idx + 1) * bs]
        start_idx += bs

        # ============ misc ... ============
        losses.update(loss.item(), inputs[0].size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        print("Epoch: [{0}][{1}]\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Lr: {lr:.4f}".format(
                    epoch,
                    it,
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    lr=optimizer.optim.param_groups[0]["lr"]
                ))
        if args.rank ==0 and it % 50 == 0:
            logger.info(
                "Epoch: [{0}][{1}]\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Lr: {lr:.4f}".format(
                    epoch,
                    it,
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    lr=optimizer.optim.param_groups[0]["lr"],
                )
            )

        if it % 50 == 0:
            if args.rank == 0:
                stats = dict(epoch=epoch, step=it,
                                 lr_weights=optimizer.param_groups[0]['lr'],
                                 loss=loss.item(),
                                 time=int(time.time() - end))
                print(json.dumps(stats))
                print(json.dumps(stats), file=stats_file)
    return (epoch, losses.avg), local_memory_index, local_memory_embeddings




if __name__== "__main__":
    parser = get_upstream_parser()
    args = parser.parse_args()

    create_dir(os.path.join(args.save_dir,'checkpoints'))
    create_dir(os.path.join(args.save_dir,'checkpoints_deepcluster'))

    args.rank = 0
    args.dist_url = 'tcp://localhost:58472'
    args.world_size = 2

    torch.multiprocessing.spawn(main, (args,), 2)

    #main(args)


















