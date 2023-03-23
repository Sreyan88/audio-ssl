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


from utils import extract_log_mel_spectrogram, compute_features, get_upstream_parser, AverageMeter, UnifLabelSampler, Logger
from specaugment import specaug
from datasets import collate_fn_padd, BARLOW
from models import AAAI_BARLOW

list_of_files_directory_1 = os.listdir("/speech/srayan/icassp/kaggle_data/audioset_train/train_wav/")
list_of_files_directory = ["/speech/srayan/icassp/kaggle_data/audioset_train/train_wav/" + item for item in list_of_files_directory_1]



AUDIO_SR = 16000
tf.config.set_visible_devices([], 'GPU')

logging.basicConfig(filename='decar.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
logger=logging.getLogger()
logger.setLevel(logging.DEBUG)

def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def main(args):
    
    torch.manual_seed(31)
    torch.cuda.manual_seed_all(31)
    np.random.seed(31)

    #list_of_files_directory = pd.read_csv(args.input)
    #list_of_files_directory = list(list_of_files_directory["files"])

    final_model = AAAI_BARLOW(args)

    final_model.model_efficient = torch.nn.DataParallel(final_model.model_efficient)

    final_model.cuda()
    logger.info(final_model)
    cudnn.benchmark = True

    optimizer = torch.optim.SGD(
        filter(lambda x: x.requires_grad, final_model.parameters()),
        lr=0.05,
        momentum=0.9,
        weight_decay=10**-5,
    )

    start_epoch = 0

    #Resume from checkpoint
    if args.resume:
        logger.info("loading checkpoint")
        checkpoint = torch.load(args.checkpoint_path)
        start_epoch = checkpoint['epoch']
        # remove top_layer parameters from checkpoint
        for key in checkpoint['state_dict'].copy():
            if 'top_layer' in key:
                del checkpoint['state_dict'][key]
        final_model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))

    train_dataset = BARLOW(list_of_files_directory)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, collate_fn = collate_fn_padd, pin_memory=True, num_workers=args.num_workers)

    best_loss = float("inf")

    for epoch in range(start_epoch,args.epochs):
    
        logger.info("Starting To Train")

        loss = train(args, train_loader, final_model, optimizer, epoch)

        logger.info("Logging and saving checkpoints")

        logger.info('###### Epoch [{0}] ###### \n'
                  'ConvNet loss: {1:.3f}'
                  .format(epoch, loss))
        
        #Save running checkpoint
        torch.save({'epoch': epoch + 1,
                    'state_dict': final_model.state_dict(),
                    'optimizer' : optimizer.state_dict()},
                    os.path.join(args.save_dir, 'checkpoints_deepcluster', 'checkpoint_' + str(epoch + 1) + "_" + '.pth.tar'))

        #Save best checkpoint
        if epoch > 0:
            if loss < best_loss:
                torch.save({'epoch': epoch + 1,
                    'state_dict': final_model.state_dict(),
                    'optimizer' : optimizer.state_dict()},
                    os.path.join(args.save_dir, 'best_loss.pth.tar'))
                best_loss = loss
        

def train(args, loader, model, opt, epoch):

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

        if n % 5000 == 0:
            logger.info('Saving Checkpoint')
            path = os.path.join(
                args.save_dir,
                'checkpoints',
                'checkpoint_' + str(n / 5000) + '.pth.tar',
            )

            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer' : opt.state_dict()
            }, path)


        input_var_1 = torch.autograd.Variable(input_tensor_1.cuda())
        input_var_2 = torch.autograd.Variable(input_tensor_2.cuda())

        loss = model(input_var_1,input_var_2)

        # record loss
        losses.update(loss, input_tensor_1.size(0))

        # compute gradient and do SGD step
        opt.zero_grad()
        loss.backward()
        opt.step()

        batch_time.update(time.time() - end)
        end = time.time()

        logger.info('Epoch: [{0}][{1}/{2}]\t'
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data: {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss: {loss.val:.4f} ({loss.avg:.4f})'
                  .format(epoch, i, len(loader), batch_time=batch_time,
                          data_time=data_time, loss=losses))

    return losses.avg


if __name__== "__main__":
    parser = get_upstream_parser()
    args = parser.parse_args()

    create_dir(os.path.join(args.save_dir,'checkpoints'))
    create_dir(os.path.join(args.save_dir,'checkpoints_deepcluster'))

    args.rank = 0
    args.dist_url = 'tcp://localhost:58472'
    args.world_size = 4

    torch.multiprocessing.spawn(main, (args,), 4)

    #main(args)


















