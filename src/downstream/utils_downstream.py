import argparse
import logging
import os
import random
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from src.downstream.CKA import kernel_CKA, linear_CKA
import sys

from src.downstream.augmentations import MixupBYOLA, RandomResizeCrop, RunningNorm

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

def get_logger(args):
    logger = logging.getLogger(__name__)
    f_handler = logging.FileHandler(os.path.join(args.exp_root,'train.log'))
    f_handler.setLevel(logging.INFO)
    logger.addHandler(f_handler)
    logger.setLevel(logging.DEBUG)
    return logger

def create_exp_dir(args):
    stats_file=None
    args.exp_root.mkdir(parents=True, exist_ok=True)
    if args.rank == 0:
        stats_file = open(args.exp_root / 'downstream_stats.txt', 'a', buffering=1)
        print(' '.join(sys.argv))
        print(' '.join(sys.argv), file=stats_file)
    return stats_file    

def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_downstream_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--down_stream_task', default="speech_commands_v1", type=str,
                        help='''down_stream task name one of
                        birdsong_freefield1010 , birdsong_warblr ,
                        speech_commands_v1 , speech_commands_v2
                        libri_100 , musical_instruments , iemocap , tut_urban , voxceleb1 , musan
                        ''')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='batch size ')
    parser.add_argument('--epochs', default=30, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--resume', default = False, type=str2bool,
                        help='number of total epochs to run')
    parser.add_argument('--pretrain_path', default=None, type=Path,
                        help='Path to Pretrain weights')
    parser.add_argument('--freeze', default=False, type=str2bool,
                        help='Path to Pretrain weights')
    parser.add_argument('--final_pooling_type', default='Avg', type=str,
                        help='valid final pooling types are Avg,Max')
    parser.add_argument('--load_only_encoder',default = True,type =str2bool)
    parser.add_argument('--tag',default = "test",type =str)
    parser.add_argument('--exp-dir',default='./exp/',type=Path,help="experiment root directory")
    parser.add_argument('--lr',default=0.001,type=float,help="experiment root directory")
    parser.add_argument('--use_model', default='effnet', type=str,
                        help='Which model to use?')
    parser.add_argument('--norm', default='l2', type=str,
                        help='Which norm to use?')
    parser.add_argument('--downstream_path', default=None, type=Path,
                        help='Path to Downstream model weights')
    parser.add_argument('--layer_prob', default=4, type=int,
                        help='probing the layer')
    parser.add_argument('--arch',default = "deloresm",type =str)
    parser.add_argument('--config', default = None,type =str)                    
    parser.add_argument('--num_workers', default = 24, type = int)
    parser.add_argument('--dataset_train_path', default = '/speech/ashish/text_downstream.csv',type =str)
    parser.add_argument('--dataset_test_path', default = '/speech/ashish/text_downstream.csv',type =str)
    return parser


def freeze_effnet(model):
    logger=logging.getLogger("__main__")
    logger.info("freezing effnet weights")
    for param in model.model_efficient.parameters():
        param.requires_grad = False

def freeze_delores(model):
    logger=logging.getLogger("__main__")
    logger.info("freezing moco weights")
    for param in model.features_1.parameters():
        param.requires_grad = False
    for param in model.features_2.parameters():
        param.requires_grad = False
    for param in model.features_3.parameters():
        param.requires_grad = False        
    for param in model.fc.parameters():
        param.requires_grad = False

def load_pretrain_effnet(path,model,
                load_only_effnet=False):
    logger=logging.getLogger("__main__")
    logger.info("loading from checkpoint only weights : "+ str(path))
    checkpoint = torch.load(path)
    if load_only_effnet :
        for key in checkpoint['state_dict'].copy():
            if not 'model_efficient' in key:
                del checkpoint['state_dict'][key]
    mod_missing_keys,mod_unexpected_keys   = model.load_state_dict(checkpoint['state_dict'],strict=False)
    logger.info("Model missing keys")
    logger.info(mod_missing_keys)
    print(mod_missing_keys)
    logger.info("Model unexpected keys")
    logger.info(mod_unexpected_keys)
    print(mod_unexpected_keys)
    logger.info("done loading")
    return model

def load_pretrain_deloresm(path,model,
                load_only_effnet=False):
    logger=logging.getLogger("__main__")
    logger.info("loading from checkpoint only weights : "+ str(path))
    backbone = Moco_v2.load_from_checkpoint(path, strict=False)
    wts = backbone.encoder_q.state_dict()
    mod_missing_keys,mod_unexpected_keys = model.module.load_state_dict(wts,strict=False)
    #checkpoint = torch.load(path)
    #if load_only_effnet :
    #    for key in checkpoint['state_dict'].copy():
    #        if ('features' not in key) and ('fc' not in key):
    #            del checkpoint['state_dict'][key]
    #mod_missing_keys,mod_unexpected_keys   = model.load_state_dict(checkpoint['state_dict'],strict=False)
    logger.info("Model missing keys")
    logger.info(mod_missing_keys)
    print(mod_missing_keys)
    logger.info("Model unexpected keys")
    logger.info(mod_unexpected_keys)
    print(mod_unexpected_keys)
    logger.info("done loading")
    return model

def load_pretrain_delores(path,model,
                load_only_effnet=False):
    logger=logging.getLogger("__main__")
    logger.info("loading from checkpoint only weights : "+ str(path))
    checkpoint = torch.load(path)
    mod_missing_keys,mod_unexpected_keys   = model.load_state_dict(checkpoint['state_dict'],strict=False)
    logger.info("Model missing keys")
    logger.info(mod_missing_keys)
    print(mod_missing_keys)
    logger.info("Model unexpected keys")
    logger.info(mod_unexpected_keys)
    print(mod_unexpected_keys)
    logger.info("done loading")
    return model

def resume_from_checkpoint(path,model,optimizer):
    logger = logging.getLogger("__main__")
    logger.info("loading from checkpoint : "+path)
    checkpoint = torch.load(path)
    start_epoch = checkpoint['epoch']
    logger.info("Task :: {}".format(checkpoint['down_stream_task']))
    mod_missing_keys,mod_unexpected_keys = model.load_state_dict(checkpoint['state_dict'],strict=False)
    opt_missing_keys,opt_unexpected_keys = optimizer.load_state_dict(checkpoint['optimizer'])
    logger.info("Model missing keys",mod_missing_keys)
    logger.info("Model unexpected keys",mod_unexpected_keys)
    logger.info("Opt missing keys",opt_missing_keys)
    logger.info("Opt unexpected keys",opt_unexpected_keys)
    logger.info("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
    return start_epoch , model, optimizer


def validate_from_checkpoint(path,model):
    logger = logging.getLogger("__main__")
    #logger.info("loading from checkpoint : "+path)
    checkpoint = torch.load(path)
    #start_epoch = checkpoint['epoch']
    #logger.info("Task :: {}".format(checkpoint['down_stream_task']))
    mod_missing_keys,mod_unexpected_keys = model.load_state_dict(checkpoint['state_dict'],strict=False)
    logger.info("Model missing keys",mod_missing_keys)
    logger.info("Model unexpected keys",mod_unexpected_keys)
    print(mod_missing_keys)
    print(mod_unexpected_keys)
    return model


def save_to_checkpoint(down_stream_task,dir,epoch,model,opt):
    torch.save({
            'down_stream_task': down_stream_task,
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer' : opt.state_dict()
            },
            os.path.join('.',dir,'models', 'checkpoint_' + str(epoch) + "_" + '.pth.tar')
    )

def set_seed(seed = 31):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def move_to_gpu(*args):
    if torch.cuda.is_available():
        for item in args:
            item.cuda()

class Metric(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val):
        if isinstance(val, (torch.Tensor)):
            val = val.numpy()
            self.val = val
            self.sum += np.sum(val)
            self.count += np.size(val)
        self.avg = self.sum / self.count

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
def check_for_freeze(model):
    for param in model.module.features_1.parameters():
        if param.requires_grad == True:
            print(param)
        else:
            print('grad is false layer 1')
            print(param)
    for param in model.module.features_2.parameters():
        if param.requires_grad == True:
            print(param)
        else:
            print('grad is false layer 2')
            print(param)    
    for param in model.module.features_3.parameters():
        if param.requires_grad == True:
            print(param)
        else:
            print('grad is false layer 3')
            print(param)            
    for param in model.module.fc.parameters():
        if param.requires_grad == True:
            print(param)
        else:
            print('grad is false layer 4')
            print(param)    
    for param in model.module.final.parameters():
        if param.requires_grad == True:
            print(param)      
    print('Every thing is fine')
def calc_norm_stats(train_dataset, test_dataset, n_stats=50000):
    """Calculates statistics of log-mel spectrogram features in a data source for normalization.
    Args:
        cfg: Configuration settings.
        data_src: Data source class object.
        n_stats: Maximum number of files to calculate statistics.
    """

    # def data_for_stats(data_src):
    #     # use all files for LOO-CV (Leave One Out CV)
    #     if data_src.loocv:
    #         return data_src
    #     # use training samples only for non-LOOCV (train/eval/test) split.
    #     return data_src.subset([0])

    # stats_data = data_for_stats(data_src)

    train_files = os.listdir(train_dataset.feat_root)
    test_files = os.listdir(test_dataset.feat_root)

    n_stats = min(n_stats, len(train_files + test_files))
    n_stats_train = int(n_stats * (len(train_files) / len(train_files + test_files)))
    n_stats_test = int(n_stats * (len(test_files) / len(train_files + test_files)))

   # logging.info(f'Calculating mean/std using random {n_stats} samples from training population {len(stats_data)} samples...')

    sample_idxes_train = np.random.choice(range(len(train_files)), size=n_stats_train, replace=False)
    sample_idxes_test = np.random.choice(range(len(test_files)), size=n_stats_test, replace=False)
    X = [train_dataset[i][0].numpy() for i in tqdm(sample_idxes_train)] + [test_dataset[i][0].numpy() for i in tqdm(sample_idxes_test)]
    X = np.hstack(X)

    norm_stats = np.array([X.mean(), X.std()])
    logging.info(f'  ==> mean/std: {norm_stats}, {norm_stats.shape} <- {X.shape}')
    return norm_stats

def bootstrap_similarity(feat1,feat2,num_samples_per_set=1000,num_sets = 50):
    all_sim = []
    size = feat1.shape[0]
    idx = np.arange(size)

    if num_samples_per_set > size:
        num_samples_per_set = size//2
    for i in range(num_sets):
        np.random.shuffle(idx)
        select_idx = idx[:num_samples_per_set]
        all_sim.append(kernel_CKA(feat1[select_idx],feat2[select_idx]))
    mean_sim = np.mean(all_sim)
    std_sim = np.std(all_sim)
    print('Mean simi: {}, std simi:{}'.format(mean_sim, std_sim))
    return np.mean(all_sim), np.std(all_sim)
