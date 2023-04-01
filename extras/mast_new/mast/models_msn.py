import re
import logging
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F

from utils import off_diagonal
from utils import trunc_normal_
from collections import OrderedDict
import argparse
import sys
from models import ASTModel
import mvit.utils.checkpoint as cu
from mvit.config.defaults import assert_and_infer_cfg, get_cfg
from mvit.utils.misc import launch_job

def parse_args():
    """
    Parse the following arguments for a default parser.
    Args:
        shard_id (int): shard id for the current machine. Starts from 0 to
            num_shards - 1. If single machine is used, then set shard id to 0.
        num_shards (int): number of shards using by the job.
        init_method (str): initialization method to launch the job with multiple
            devices. Options includes TCP or shared file-system for
            initialization. details can be find in
            https://pytorch.org/docs/stable/distributed.html#tcp-initialization
        cfg (str): path to the config file.
        opts (argument): provide addtional options from the command line, it
            overwrites the config loaded from file.
    """
    parser = argparse.ArgumentParser(
        description="Provide training and testing pipeline."
    )
    parser.add_argument(
        "--shard_id",
        help="The shard id of current node, Starts from 0 to num_shards - 1",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--num_shards",
        help="Number of shards using by the job",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--init_method",
        help="Initialization method, includes TCP or shared file-system",
        default="tcp://localhost:9999",
        type=str,
    )
    parser.add_argument(
        "--cfg",
        dest="cfg_file",
        help="Path to the config file",
        default="configs/MVITv2_B.yaml",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="See mvit/config/defaults.py for all options",
        default=None,
        nargs=argparse.REMAINDER,
    )
    if len(sys.argv) == 1:
        parser.print_help()
    return parser.parse_args()


def load_config(args):
    """
    Given the arguemnts, load and initialize the configs.
    Args:
        args (argument): arguments includes `shard_id`, `num_shards`,
            `init_method`, `cfg_file`, and `opts`.
    """
    # Setup cfg.
    cfg = get_cfg()
    # Load config from cfg.
    if args.cfg_file is not None:
        cfg.merge_from_file(args.cfg_file)
    # Load config from command line, overwrite config from opts.
    if args.opts is not None:
        cfg.merge_from_list(args.opts)

    # Inherit parameters from args.
    if hasattr(args, "num_shards") and hasattr(args, "shard_id"):
        cfg.NUM_SHARDS = args.num_shards
        cfg.SHARD_ID = args.shard_id
    if hasattr(args, "rng_seed"):
        cfg.RNG_SEED = args.rng_seed
    if hasattr(args, "output_dir"):
        cfg.OUTPUT_DIR = args.output_dir

    # Create the checkpoint dir.
    cu.make_checkpoint_dir(cfg.OUTPUT_DIR)
    return cfg





class NetworkCommonMixIn():
    """Common mixin for network definition."""

    def load_weight(self, weight_file, device):
        """Utility to load a weight file to a device."""

        state_dict = torch.load(weight_file, map_location=device)
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        # Remove unneeded prefixes from the keys of parameters.
        weights = {}
        for k in state_dict:
            m = re.search(r'(^fc\.|\.fc\.|^features\.|\.features\.)', k)
            if m is None: continue
            new_k = k[m.start():]
            new_k = new_k[1:] if new_k[0] == '.' else new_k
            weights[new_k] = state_dict[k]
        # Load weights and set model to eval().
        self.load_state_dict(weights)
        self.eval()
        logging.info(f'Using audio embbeding network pretrained weight: {Path(weight_file).name}')
        return self

    def set_trainable(self, trainable=False):
        for p in self.parameters():
            p.requires_grad = trainable


args = parse_args()
cfg = load_config(args)
cfg = assert_and_infer_cfg(cfg)

class AudioNTT2020(nn.Module):
    """BYOL-A General Purpose Representation Network.
    This is an extension of the DCASE 2020 Task 6 NTT Solution Audio Embedding Network.
    """

    def __init__(self, out_dim, use_bn = False, norm_last_layer=True, n_layers=3, hidden_dim=512, n_mels=64, d=768, output_dim=256):
        #super().__init__(n_mels=n_mels, d=d)
        super(AudioNTT2020, self).__init__()
        #self.args = args
        #self.units = args.final_units
        self.ast_model = ASTModel(cfg, label_dim=521, input_tdim=1024, imagenet_pretrain = False, audioset_pretrain = False, model_size='mvit')

        # layers = [nn.Linear(d, hidden_dim)]
        # if use_bn:
        #     layers.append(nn.BatchNorm1d(2048))
        # layers.append(nn.GELU())
        # for _ in range(n_layers - 2):
        #     layers.append(nn.Linear(hidden_dim, hidden_dim))
        #     if use_bn:
        #         layers.append(nn.BatchNorm1d(hidden_dim))
        #     layers.append(nn.GELU())
        # layers.append(nn.Linear(hidden_dim, bottleneck_dim))
        # self.mlp = nn.Sequential(*layers)
        # self.apply(self._init_weights)
        # self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        # self.last_layer.weight_g.data.fill_(1)
        # if norm_last_layer:
        #     self.last_layer.weight_g.requires_grad = False

        emb_dim = d
        fc = OrderedDict([])
        fc['fc1'] = torch.nn.Linear(emb_dim, output_dim)
        #if True: #use_bn = True
        #    fc['bn1'] = torch.nn.BatchNorm1d(hidden_dim)
        #fc['gelu1'] = torch.nn.GELU()
        self.mlp = torch.nn.Sequential(fc)

    # def _init_weights(self, m):
    #     if isinstance(m, nn.Linear):
    #         trunc_normal_(m.weight, std=.02)
    #         if isinstance(m, nn.Linear) and m.bias is not None:
    #             nn.init.constant_(m.bias, 0)

    def forward(self, batch, return_before_head=False):

        z = self.ast_model(batch, patch_drop=0.0)
        x = self.mlp(z.float())
        #print('x', x)
        #if return_before_head == True:
        #    return x,x
        # x = nn.functional.normalize(x, dim=-1, p=2)
        # x = self.last_layer(x)

        return x
