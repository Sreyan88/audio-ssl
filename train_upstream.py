import os
import sys
import torch
import yaml
import argparse
import importlib
import torch.nn.functional as F
import torch.nn as nn
import torch.distributed as dist
import pandas as pd
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from src.dataset import BaselineDataModule
from src.augmentations import AugmentationModule

def main(args):

    list_of_files_directory = pd.read_csv(args.input)

    if args.config is None:
        default_upstream_config = "src/upstream/" + args.upstream + "/config.yaml"
        with open(default_upstream_config, 'r') as duc:
            config = yaml.load(duc, Loader=yaml.FullLoader)
    else:
        with open(args.config, 'r') as duc:
            config = yaml.load(duc, Loader=yaml.FullLoader)
    print(config)

    # @Ashish waht is this? Do we require it? If not lets remove, its not training when I add this, getting stuck
    # dist.init_process_group('gloo', init_method='file:///tmp/somefile', rank=0, world_size=config["run"]["world_size"])

    # load augmentation module
    tfms = AugmentationModule(config, len(list_of_files_directory))

    if args.upstream == "unfused":
        labels = list(list_of_files_directory["label"])
        list_of_files_directory = list(list_of_files_directory["files"])
        dm = BaselineDataModule(args, tfms, train_data_dir_list = list_of_files_directory,labels=labels,num_workers=config["run"]["num_dataloader_workers"],batch_size=config["run"]["batch_size"])

    else:
        list_of_files_directory = list(list_of_files_directory["files"])
        dm = BaselineDataModule(config, tfms, train_data_dir_list = list_of_files_directory,num_workers=config["run"]["num_dataloader_workers"],batch_size=config["run"]["batch_size"]) 
    
    # load upstream expert
    module_path_expert = f'src.upstream.{args.upstream}.upstream_expert'
    expert = getattr(importlib.import_module(module_path_expert), 'Upstream_Expert')

    # load base encoder
    module_path_base_encoder = f'src.encoder'
    base_encoder = getattr(importlib.import_module(module_path_base_encoder), config["pretrain"]["base_encoder"]["type"])
    
    model = expert(config, base_encoder=base_encoder, datamodule=dm)

    # @Ashish do we need lambda for every term, please check this and modularize, 
    # add to conf, also please remove from dir_path, also remove from parser
    # lamb_append_term = '-'.join(np.array(args.lamb_values).astype(str))
    
    checkpoint_callback = ModelCheckpoint(
                                dirpath=config["run"]["save_path"]+'chkp_{0}'.format(config["pretrain"]["base_encoder"]),
                                filename='{epoch}-{val_loss:.3f}',
                                monitor="train_loss", 
                                mode="min",
                                save_top_k=3)
        
    if torch.cuda.is_available():
        if args.load_checkpoint:
            trainer = pl.Trainer(gpus=config["run"]["world_size"], callbacks = [checkpoint_callback], accelerator="gpu", strategy="ddp", resume_from_checkpoint=args.load_checkpoint)
        else:
            trainer = pl.Trainer(gpus=config["run"]["world_size"], callbacks = [checkpoint_callback],accelerator="gpu", strategy="ddp")
    else:
        trainer = pl.Trainer(checkpoint_callback = checkpoint_callback,)
    
    trainer.fit(model, dm)


def get_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)

    # Clean the ones not required @Ashish

    # Add data arguments
    parser.add_argument("--input", help="path to data directory", type=str, default='data/pre_train.csv')
    parser.add_argument("--batch-size", default=16, type=int, help="train batch size")
    parser.add_argument("--lamb_values",nargs="+",help="my help message", type=float, default= [5e-5,5e-5,5e-5,5e-5])
    parser.add_argument("--save-path", help="path to saving model directory", type=str, default="./")
    parser.add_argument('--load_checkpoint', type=str, help='load checkpoint', default = None)
    parser.add_argument('-c', '--config', metavar='CONFIG_PATH', help='The yaml file for configuring the whole experiment, except the upstream model', default = None)
    parser.add_argument('--upstream', type=str, help='define the type of upstream', default = 'unfused')
    # Add model arguments
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    main(args)