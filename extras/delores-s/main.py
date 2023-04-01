import argparse
import os
import pickle
import time
import numpy as np
import torch.backends.cudnn as cudnn
from sklearn.metrics.cluster import normalized_mutual_info_score
from os.path import join as path_join
import json
import torch
import tensorflow as tf
import logging
from torch import nn
import pandas as pd


from utils import extract_log_mel_spectrogram, compute_features, get_upstream_parser, AverageMeter, UnifLabelSampler, Logger
from specaugment import specaug
from datasets import collate_fn_padd, BARLOW
from models import AAAI_BARLOW
from multi_proc import LARS, adjust_learning_rate
from augmentations import MixupBYOLA, RandomResizeCrop, RunningNorm
from models_byol import AudioNTT2020

#list_of_files_directory_1 = os.listdir("/speech/srayan/icassp/kaggle_data/audioset_train/train_wav/")
#list_of_files_directory = ["/speech/srayan/icassp/kaggle_data/audioset_train/train_wav/" + item for item in list_of_files_directory_1]

AUDIO_SR = 16000
tf.config.set_visible_devices([], 'GPU')

logging.basicConfig(filename='decar_l2_byol.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
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
        return self.train_transform(x), self.train_transform(x)

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

    if args.use_model == 'effnet':
        model = AAAI_BARLOW(args).cuda(gpu)
        print('in effnet')
    elif args.use_model == 'byol':
        model = AudioNTT2020(args, n_mels=64, d=2048).cuda(gpu)

    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    param_biases = []
    param_weights = []
    for param in model.parameters():
        if param.ndim == 1:
            param_biases.append(param)
        else:
            param_weights.append(param)

    parameters = [{'params': param_weights}, {'params': param_biases}]
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    optimizer = LARS(parameters, lr=0, weight_decay=1e-6,
                     weight_decay_filter=True,
                     lars_adaptation_filter=True)

    tfms = AugmentationModule(args, (64, 96), 2 * len(list_of_files_directory))
    train_dataset = BARLOW(args, list_of_files_directory, tfms)
    sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

    per_device_batch_size = args.batch_size // args.world_size

    loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=per_device_batch_size, num_workers=args.num_workers,
        pin_memory=True, sampler=sampler)

    logger.info(model)

    start_epoch = 0

    #Resume from checkpoint
    if args.resume:
        logger.info("loading checkpoint")
        checkpoint = torch.load(args.checkpoint_path)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))

    best_loss = float("inf")
    scaler = torch.cuda.amp.GradScaler()

    if args.rank == 0:
        stats_file = open(args.save_dir  + "/" + 'stats.txt', 'a', buffering=1)
    else:
        stats_file = None

    for epoch in range(start_epoch,args.epochs):

        sampler.set_epoch(epoch)
    
        if args.rank == 0:
            logger.info("Starting To Train")

        loss = train(args, loader, model, optimizer, epoch, gpu, scaler, stats_file)

        if args.rank == 0:
            logger.info("Logging and saving checkpoints")

            logger.info('###### Epoch [{0}] ###### \n'
                    'ConvNet loss: {1:.3f}'
                    .format(epoch, loss))
        
        #Save running checkpoint

        if args.rank == 0:

            torch.save({'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'optimizer' : optimizer.state_dict()},
                        os.path.join(args.save_dir, 'checkpoints_deepcluster', 'checkpoint_' + str(epoch + 1) + "_" + '.pth.tar'))

            #Save best checkpoint
            if loss < best_loss:
                torch.save({'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict()},
                    os.path.join(args.save_dir, 'best_loss.pth.tar'))
                best_loss = loss
        

def train(args, loader, model, optimizer, epoch, gpu, scaler, stats_file):

    batch_time = AverageMeter()
    losses = AverageMeter()
    data_time = AverageMeter()
    forward_time = AverageMeter()
    backward_time = AverageMeter()

    model.train()

    end = time.time()
    
    for i, (input_tensor_1,input_tensor_2) in enumerate(loader):
        data_time.update(time.time() - end)

        n = len(loader) * epoch + i

        if n % 500 == 0:
            if args.rank == 0:
                logger.info('Saving Checkpoint')
                path = os.path.join(
                    args.save_dir,
                    'checkpoints',
                    'checkpoint_' + str(n / 500) + '.pth.tar',
                )

                torch.save({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict()
                }, path)

        input_var_1 = input_tensor_1.cuda(gpu, non_blocking=True)
        input_var_2 = input_tensor_2.cuda(gpu, non_blocking=True)

        adjust_learning_rate(args, optimizer, loader, i)
        optimizer.zero_grad()

        #print(input_var_1.shape)
        #print(input_var_2.shape)
        with torch.cuda.amp.autocast():
            loss = model.forward(input_var_1, input_var_2)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # record loss
        losses.update(loss, input_tensor_1.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if args.rank == 0:
            logger.info('Epoch: [{0}][{1}/{2}]\t'
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data: {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss: {loss.val:.4f} ({loss.avg:.4f})'
                    .format(epoch, i, len(loader), batch_time=batch_time,
                            data_time=data_time, loss=losses))

            stats = dict(epoch=epoch, step=n, loss=loss.item())
            print(json.dumps(stats))
            print(json.dumps(stats), file=stats_file)

    return losses.avg


if __name__== "__main__":
    parser = get_upstream_parser()
    args = parser.parse_args()

    create_dir(os.path.join(args.save_dir,'checkpoints'))
    create_dir(os.path.join(args.save_dir,'checkpoints_deepcluster'))

    args.rank = 0
    args.dist_url = 'tcp://localhost:58467'
    args.world_size = 1

    torch.multiprocessing.spawn(main, (args,), 1)

    #main(args)


















