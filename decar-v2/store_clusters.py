import argparse
from operator import index
import os
import pickle
import time
import numpy as np
import torch.backends.cudnn as cudnn
from sklearn.metrics.cluster import normalized_mutual_info_score
from os.path import join as path_join
import json
import torch
#import tensorflow as tf
import logging
from torch import nn


from utils import extract_log_mel_spectrogram, compute_features, get_upstream_parser, AverageMeter, UnifLabelSampler, Logger
from clustering import run_kmeans, Kmeans, PIC, rearrange_clusters
from specaugment import specaug
from datasets import DeepCluster, DeepCluster_Reassigned
from models_delores import AudioNTT2020
from specaugment import specaug
from augmentations import MixupBYOLA, RandomResizeCrop, RunningNorm
import pandas as pd

AUDIO_SR = 16000


logging.basicConfig(filename='decar.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
logger=logging.getLogger()
logger.setLevel(logging.DEBUG)

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
        return self.train_transform(x)


def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def make_dataset(audio_files, audio_labels, audio_indexes):
    label_to_idx = {label: idx for idx, label in enumerate(set(audio_labels))}
    audiopath_w_labels = []
    for i, index in enumerate(audio_indexes):
        path = audio_files[index]
        pseudolabel = label_to_idx[audio_labels[index]] #could have been pseudolabels, bekar confusion change later
        audiopath_w_labels.append((path,pseudolabel))
            
    return audiopath_w_labels

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

    list_of_files_directory = pd.read_csv('/nlsasfs/home/nltm-pilot/ashishs/libri100/train_data.csv')
    list_of_files_directory = list(list_of_files_directory["AudioPath"])
    list_of_files_directory = ['/nlsasfs/home/nltm-pilot/ashishs/libri100/wav/'+x for x in list_of_files_directory]
    #final_model = DeepCluster_ICASSP()
    #final_model.model_efficient = torch.nn.DataParallel(final_model.model_efficient)
    print('model defination')
    final_model = AudioNTT2020(args, args.feat_dim, n_mels=64, d=2048).cuda(gpu)
    #print(final_model.classifier)
    #fd = int(final_model.top_layer.weight.size()[1])
    #final_model.top_layer = None
    final_model = nn.SyncBatchNorm.convert_sync_batchnorm(final_model)

    final_model = nn.parallel.DistributedDataParallel(final_model, device_ids=[gpu],find_unused_parameters=True)
    print('initilization of the model completed')
    print(final_model)
    
    
    #final_model.cuda()
    logger.info(final_model)
    cudnn.benchmark = True


    #optimizer = torch.optim.SGD(
    #    filter(lambda x: x.requires_grad, final_model.parameters()),
    #    lr=0.05,
    #    momentum=0.9,
    #    weight_decay=10**-5,
    #)

    criterion = nn.CrossEntropyLoss().cuda()

    start_epoch = 0

    #Resume from checkpoint
    if True:
        logger.info("loading checkpoint")
        checkpoint = torch.load('/nlsasfs/home/nltm-pilot/ashishs/dump_all_model/apex_changedlr_indentify_checkpoints_upstream_deepcluster_new_norm_changed/checkpoints_deepcluster/checkpoint_89_.pth.tar')
        start_epoch = checkpoint['epoch']
        # remove top_layer parameters from checkpoint
        for key in checkpoint['state_dict'].copy():
            if 'top_layer' in key:
                del checkpoint['state_dict'][key]
        final_model.load_state_dict(checkpoint['state_dict'])
        #optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))

    cluster_log = Logger(os.path.join(args.save_dir, 'clusters'))

    pretrain_dataset = DeepCluster(list_of_files_directory,len(list_of_files_directory),args)  #without augmentation 

    train_loader = torch.utils.data.DataLoader(pretrain_dataset, batch_size=args.batch_size, num_workers=16)

    best_loss = float("inf")
    tfms = AugmentationModule(args, (64, 97), 2 * len(list_of_files_directory))
    #final_model.module.top_layer = None
    #final_model.module.classifier = nn.Sequential(*list(final_model.module.classifier.children())[:-1])

    features, times = compute_features(args, train_loader, final_model, len(list_of_files_directory))
    logger.info("Entering Clustering")
    deepcluster = Kmeans(585)
    clustering_loss = deepcluster.cluster(features, verbose=True)
    #mlp = list(final_model.module.classifier.children())
    #mlp.append(nn.ReLU(inplace=True).cuda())
    #final_model.module.classifier = nn.Sequential(*mlp)
    #final_model.module.top_layer = nn.Linear(fd, len(deepcluster.images_lists))
    #final_model.module.top_layer.weight.data.normal_(0, 0.01)
    #final_model.module.top_layer.bias.data.zero_()
    #final_model.module.top_layer.cuda()

    logger.info("Starting To make Reassigned Dataset")

    pseudolabels = []
    image_indexes = []
    for cluster, images in enumerate(deepcluster.images_lists):
        image_indexes.extend(images)
        pseudolabels.extend([cluster] * len(images))

    indexes_sorted = np.argsort(image_indexes)  
    pseudolabels = np.asarray(pseudolabels)[indexes_sorted]
    dataset_w_labels = make_dataset(list_of_files_directory,pseudolabels,indexes_sorted)

    with open('./libri_100_new.csv','w') as f:
        for x in dataset_w_labels:
            f.write(x[0]+','+str(x[1])+'\n')

if __name__== "__main__":
    parser = get_upstream_parser()
    args = parser.parse_args()

    create_dir(os.path.join(args.save_dir,'checkpoints'))
    create_dir(os.path.join(args.save_dir,'checkpoints_deepcluster'))

    args.rank = 0
    args.dist_url = 'tcp://localhost:58472'
    args.world_size = 1

    torch.multiprocessing.spawn(main, (args,), 1)            
