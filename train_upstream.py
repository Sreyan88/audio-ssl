import os
import sys
import torch
import argparse
import torch.nn.functional as F
import torch.nn as nn
import torch.distributed as dist
import pandas as pd
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from src.dataset import BaselineDataModule
from utils import AugmentationModule

def main(args):

    dist.init_process_group('gloo', init_method='file:///tmp/somefile', rank=0, world_size=1)

    list_of_files_directory = pd.read_csv(args.input)

    if args.config is None:
        with open(args.config, 'r') as file:
            config = yaml.load(put_a_default_here, Loader=yaml.FullLoader)
    else:
        with open(args.config, 'r') as file:
            config = yaml.load(file, Loader=yaml.FullLoader)

    if args.upstream == "unfused":
        labels = list(list_of_files_directory["label"])
        list_of_files_directory = list(list_of_files_directory["files"])
        dm = BaselineDataModule(args, tfms, train_data_dir_list = list_of_files_directory,labels=labels,num_workers=config["run"]["num_dataloader_workers"],batch_size=["run"]["batch_size"])

    else:
        list_of_files_directory = list(list_of_files_directory["files"])
        dm = BaselineDataModule(args, tfms, train_data_dir_list = list_of_files_directory,num_workers=config["run"]["num_dataloader_workers"],batch_size=["run"]["batch_size"]) 


    tfms = AugmentationModule(args, (64, 96), 2 * len(list_of_files_directory)) #Ashish why 2*, please write logic
    
    module_path = f'src.upstream.{self.args.upstream}.upstream_expert'
    expert = getattr(importlib.import_module(module_path), 'Upstream_Expert')
    
    model = expert(args, datamodule=dm, lamb_values = config["pretrain"][])

    lamb_append_term = '-'.join(np.array(args.lamb_values).astype(str))
    
    checkpoint_callback = ModelCheckpoint(
                                dirpath=args.save_path+'chkp_{0}'.format(args.model_type)+lamb_append_term+'/',
                                filename='{epoch}-{val_loss:.3f}',
                                monitor="train_loss", 
                                mode="min",
                                save_top_k=3)
        
    if torch.cuda.is_available():
        if args.load_checkpoint:
            trainer = pl.Trainer(gpus=1,checkpoint_callback = checkpoint_callback,accelerator="ddp",resume_from_checkpoint=args.load_checkpoint)
        else:
            trainer = pl.Trainer(gpus=1,checkpoint_callback = checkpoint_callback,accelerator="ddp")
    else:
        trainer = pl.Trainer(checkpoint_callback = checkpoint_callback,)
    trainer.fit(model, dm)


def get_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)

    # Add data arguments
    parser.add_argument("--input", help="path to data directory", type=str, default='/nlsasfs/home/nltm-pilot/ashishs/DECAR/libri_100_new.csv')
    parser.add_argument("--batch-size", default=16, type=int, help="train batch size")
    parser.add_argument("--lamb_values",nargs="+",help="my help message", type=float, default= [5e-5,5e-5,5e-5,5e-5])
    parser.add_argument("--save-path", help="path to saving model directory", type=str, default="./")
    parser.add_argument("--use_norm", help="type of norm to be used", type=str,default= "byol")
    parser.add_argument('--length_wave', type=float, help='Length of wave split', default = 0.95)
    parser.add_argument('--load_checkpoint', type=str, help='load checkpoint', default = None)
    parser.add_argument('-c', '--config', metavar='CONFIG_PATH', help='The yaml file for configuring the whole experiment, except the upstream model')
    parser.add_argument('--upstream', type=str, help='define the type of upstream', default = 'unfused')
    # Add model arguments
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    main(args)