import os
import sys
import torch
import argparse
import torch.nn.functional as F
import torch.nn as nn
import pandas as pd
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from moco_dataset import BaselineDataModule
from moco_model import Moco_v2
from augmentations import MixupBYOLA, RandomResizeCrop, RunningNorm
import torch.distributed as dist
#list_of_files_directory_1 = os.listdir("/speech/srayan/icassp/kaggle_data/audioset_train/train_wav/")
#list_of_files_directory = ["/speech/srayan/icassp/kaggle_data/audioset_train/train_wav/" + item for item in list_of_files_directory_1]
dist.init_process_group('gloo', init_method='file:///tmp/somefile', rank=0, world_size=1)
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


def main(args):

    list_of_files_directory = pd.read_csv(args.input)
    list_of_files_directory = list(list_of_files_directory["files"])

    tfms = AugmentationModule(args, (64, 96), 2 * len(list_of_files_directory))

    dm = BaselineDataModule(args, tfms, train_data_dir_list = list_of_files_directory,num_workers=40,batch_size=args.batch_size) 
    
    # dm.train_transforms = Moco2TrainTransformsDR(height=299)
    # dm.val_transforms = Moco2EvalTransformsDR(height=299)
    # dm.test_transforms = Moco2EvalTransformsDR(height=299)
    
    model = Moco_v2(args, datamodule=dm, lamb_values = args.lamb_values)
    #if args.load_checkpoint:
    #    model = model.load_from_checkpoint(checkpoint_path=args.load_checkpoint)
    lamb_append_term = '-'.join(np.array(args.lamb_values).astype(str))
    
    checkpoint_callback = ModelCheckpoint(
                                dirpath=args.save_path+'mocov2_all_augs_dr_bt_mod_'+lamb_append_term+'/',
                                filename='{epoch}-{val_loss:.3f}',
                                monitor="train_loss", 
                                mode="min",
                                save_top_k=3)
        
    if torch.cuda.is_available():
        if args.load_checkpoint:
            trainer = pl.Trainer(gpus=1,checkpoint_callback = checkpoint_callback,accelerator="ddp",resume_from_checkpoint=args.load_checkpoint)
        else:
            trainer = pl.Trainer(gpus=1,checkpoint_callback = checkpoint_callback,accelerator="ddp")
        #,accelerator="ddp"
    else:
        trainer = pl.Trainer(checkpoint_callback = checkpoint_callback,)
    trainer.fit(model, dm)


def get_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)

    # Add data arguments
    parser.add_argument("--input", help="path to data directory", type=str, required=False)
    parser.add_argument("--batch-size", default=16, type=int, help="train batch size")
    parser.add_argument("--lamb_values",nargs="+",help="my help message", type=float, default= [5e-5,5e-5,5e-5,5e-5])
    parser.add_argument("--save-path", help="path to saving model directory", type=str, default="./")
    parser.add_argument("--use_norm", help="type of norm to be used", type=str,default= "byol")
    parser.add_argument('--length_wave', type=float, help='Length of wave split', default = 0.95)
    parser.add_argument('--load_checkpoint', type=str, help='load checkpoint', default = None)
    # Add model arguments
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    main(args)
