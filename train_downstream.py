import logging
import os
import time
import json
import matplotlib.pyplot as plt
import torch
from torch import nn
import sys
import importlib
import yaml

from src.augmentations import AugmentationModule
from src.utils import check_downstream_hf_availability
from src.downstream.downstream_encoder import DownstreamEncoder
from src.downstream.downstream_dataset import DownstreamDatase,DownstreamDatasetHF
from src.downstream.utils_downstream import (AverageMeter, get_logger, create_exp_dir, \
Metric, freeze_effnet, freeze_delores, get_downstream_parser, load_pretrain_effnet, load_pretrain_deloresm, load_pretrain_delores)


def main(gpu, args):

    if args.config is None:
        default_upstream_config = "src/downstream/downstream_config.yaml"
        with open(default_upstream_config, 'r') as duc:
            config = yaml.load(duc, Loader=yaml.FullLoader)
    else:
        with open(args.config, 'r') as duc:
            config = yaml.load(duc, Loader=yaml.FullLoader)
    print(config)

    args.rank += gpu
    torch.distributed.init_process_group(
        backend='nccl', init_method=args.dist_url,
        world_size=args.world_size, rank=args.rank)
    stats_file=None
    args.exp_root = args.exp_dir / args.tag

    stats_file=create_exp_dir(args)

    args.exp_root.mkdir(parents=True, exist_ok=True)
    if args.rank == 0:
        stats_file = open(args.exp_root / 'downstream_stats.txt', 'a', buffering=1)
        print(' '.join(sys.argv))
        print(' '.join(sys.argv), file=stats_file)
    logger = get_logger(args)
    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = True # ! change it set seed

    
    assert config['run']['batch_size'] % args.world_size == 0
    per_device_batch_size = config['run']['batch_size'] // args.world_size

    # If the dataset is availble in HuggingFace
    if check_downstream_hf_availability(args.task) == "hf":
        train_dataset = DownstreamDatasetHF(args,config,split='train')
        test_dataset = DownstreamDatasetHF(args,config,split='test')
        if args.valid_csv:
            eval_dataset = DownstreamDatasetHF(args,config,split='valid')
    # If the dataset is NOT availble in HuggingFace
    else:
        train_dataset = DownstreamDataset(args,config,split='train')
        test_dataset = DownstreamDataset(args,config,split='test',labels_dict=train_dataset.labels_dict)
        if args.valid_csv:
            eval_dataset = DownstreamDataset(args,config,split='valid',labels_dict=train_dataset.labels_dict)
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True, seed=1) #shuffle

    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=per_device_batch_size,
                                                pin_memory=True,sampler = train_sampler,num_workers=24)

    test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=args.batch_size,
                                                pin_memory=True, num_workers=24)

    #load base encoder
    module_path_base_encoder = f'src.encoder'
    base_encoder = getattr(importlib.import_module(module_path_base_encoder), config["downstream"]["base_encoder"]["type"])
    model = DownstreamEncoder(config, args, base_encoder, no_of_classes=train_dataset.self.no_of_classes).cuda(gpu) # need to get it from get_data function or somewhere else. 
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if args.freeze:
        if config["downstream"]["base_encoder"]["type"] == "effnet":
            freeze_effnet(model)
        elif config["downstream"]["base_encoder"]["type"] == "audiontt":
            freeze_delores(model)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

    criterion = nn.CrossEntropyLoss().cuda(gpu)
    optimizer = torch.optim.Adam(
        filter(lambda x: x.requires_grad, model.parameters()),
        lr=args.lr,
    )

    if args.rank == 0 : logger.info("started training")

    train_accuracy = []
    train_losses=[]
    test_accuracy = []
    test_losses=[]

    for epoch in range(0, args.epochs):
        train_sampler.set_epoch(epoch)
        train_stats = train_one_epoch(train_loader, model, criterion, optimizer, epoch,gpu,args)

        if args.rank == 0 :
            eval_stats = eval(epoch,model,test_loader,criterion,gpu)
            test_accuracy.append(eval_stats["accuracy"].avg)
            stats = dict(epoch=epoch,
                    Train_loss=train_stats["loss"].avg.cpu().numpy().item(),
                    Test_Loss=(eval_stats["loss"].avg).numpy().item(),
                    Test_Accuracy =eval_stats["accuracy"].avg,
                    Best_Test_Acc=max(test_accuracy))
            print(stats)
            print(json.dumps(stats), file=stats_file)
    
    if args.rank ==0 :
        print("max valid accuracy : {}".format(max(test_accuracy)))
        plt.plot(range(1,len(train_accuracy)+1), train_accuracy, label = "train accuracy",marker = 'x')
        plt.legend()
        plt.savefig(args.exp_root / 'accuracy.png')


def train_one_epoch(loader, model, crit, opt, epoch,gpu,args):
    '''
    Train one Epoch
    '''
    logger = logging.getLogger(__name__)
    logger.debug("epoch:"+str(epoch) +" Started")
    batch_time = AverageMeter()
    losses = AverageMeter()
    data_time = AverageMeter()

    model.train() # ! imp
    end = time.time()
    for i, (input_tensor, target) in enumerate(loader):
        data_time.update(time.time() - end)

        output = model(input_tensor.cuda(gpu, non_blocking=True))
        loss = crit(output, target.cuda(gpu, non_blocking=True))

        losses.update(loss.data, input_tensor.size(0))
        opt.zero_grad()
        loss.backward()
        opt.step()

        batch_time.update(time.time() - end)
        end = time.time()
        if args.rank ==0 :
            print('Epoch: [{0}][{1}/{2}]\t'
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data: {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss: {loss.val:.4f} ({loss.avg:.4f})'
                    .format(epoch, i, len(loader), batch_time=batch_time,
                            data_time=data_time, loss=losses))


    logger.debug("epoch-"+str(epoch) +" ended")
    stats = dict(epoch=epoch,loss=losses)
    return stats

@torch.no_grad()
def eval(epoch,model,loader,crit,gpu):
    model.eval() # ! Imp
    losses = AverageMeter()
    accuracy = Metric()
    with torch.no_grad():
        for step, (input_tensor, targets) in enumerate(loader):
           # input_tensor = torch.squeeze(input_tensor,0)
            if torch.cuda.is_available():
                input_tensor =input_tensor.cuda(gpu ,non_blocking=True)
                targets = targets.cuda(gpu,non_blocking=True)
            with torch.cuda.amp.autocast():
                outputs = model(input_tensor)
                loss = crit(outputs, targets)
                preds = torch.argmax(outputs,dim=1)==targets

            accuracy.update(preds.cpu())
            losses.update(loss.cpu().data, input_tensor.size(0))

    stats = dict(epoch=epoch,loss=losses, accuracy = accuracy)
    return stats

def get_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)

    # Clean the ones not required @Ashish

    # Add data arguments
    parser.add_argument("--task", help="path to data directory", type=str, default='speech_commands_v1')
    parser.add_argument("--train_csv", help="path to data directory", type=str, default='/speech/ashish/test_audio_label.csv')
    parser.add_argument("--valid_csv", help="path to data directory", type=str, default='/speech/ashish/test_audio_label.csv')
    parser.add_argument("--test_csv", help="path to data directory", type=str, default='/speech/ashish/test_audio_label.csv')
    parser.add_argument('--load_checkpoint', type=str, help='load checkpoint', default = None)
    parser.add_argument('-c', '--config', metavar='CONFIG_PATH', help='The yaml file for configuring the whole experiment, except the upstream model', default = None)
    # Add model arguments
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    args.ngpus_per_node = torch.cuda.device_count()
    args.rank = 0
    args.dist_url = 'tcp://localhost:58367'
    args.world_size = args.ngpus_per_node
    torch.multiprocessing.spawn(main, (args,), args.ngpus_per_node)
