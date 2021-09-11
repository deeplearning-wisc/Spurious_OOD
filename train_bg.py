import argparse
import os
import time
import random
import logging
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from models.resnet import load_model
from utils import AverageMeter, save_checkpoint, accuracy
from datasets.color_mnist import get_biased_mnist_dataloader
from datasets.cub_dataset import get_waterbird_dataloader
from datasets.celebA_dataset import get_celebA_dataloader
import math

parser = argparse.ArgumentParser(description=' use resnet (pretrained)')

parser.add_argument('--in-dataset', default="celebA", type=str, choices = ['celebA', 'color_mnist', 'waterbird'], help='in-distribution dataset e.g. IN-9')
parser.add_argument('--model-arch', default='resnet18', type=str, help='model architecture e.g. resnet50')
parser.add_argument('--domain-num', default=4, type=int,
                    help='the number of environments for model training')
parser.add_argument('--method', default='erm', type=str, help='method used for model training')
parser.add_argument('--save-epoch', default=5, type=int,
                    help='save the model every save_epoch, default = 10') 
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    help='print frequency (default: 10)')
# ID train & val batch size
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    help='mini-batch size (default: 64) used for training')
# training schedule
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--epochs', default=30, type=int,
                    help='number of total epochs to run, default = 30')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    help='initial learning rate')
parser.add_argument('--num-classes', default=2, type=int,
                    help='number of classes for model training')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', '--wd', default=0.005, type=float,
                    help='weight decay (default: 0.0001)')
parser.add_argument('--data_label_correlation', default=0.9, type=float,
                    help='data_label_correlation')
# saving, naming and logging
parser.add_argument('--exp-name', default = 'erm_new_0.9', type=str, 
                    help='help identify checkpoint')
parser.add_argument('--name', default="erm_rebuttal", type=str,
                    help='name of experiment')
parser.add_argument('--log_name', type = str, default = "info.log",
                    help='Name of the Log File')
# Device options
parser.add_argument('--gpu-ids', default='6', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--local_rank', default=-1, type=int,
                        help='rank for the current node')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--n-g-nets', default=1, type=int,
                    help="the number of networks for g_model in ReBias")
parser.add_argument('--penalty-multiplier', default=1.1, type=float,
                    help="the penalty multiplier used in IRM training")

parser.add_argument('--cosine', action='store_false',
                        help='using cosine annealing')
parser.add_argument('--lr_decay_epochs', type=str, default='15,25',
                        help=' 15, 25, 40 for waterbibrds; 10, 15 ,20 for color_mnist')
parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')

args = parser.parse_args()

state = {k: v for k, v in args._get_kwargs()}

directory = "checkpoints/{in_dataset}/{name}/{exp}/".format(in_dataset=args.in_dataset, 
            name=args.name, exp=args.exp_name)
os.makedirs(directory, exist_ok=True)
save_state_file = os.path.join(directory, 'args.txt')
fw = open(save_state_file, 'w')
print(state, file=fw)
fw.close()

# CUDA Specification
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
if torch.cuda.is_available():
    torch.cuda.set_device(args.local_rank)
device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

# Set random seed
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
set_random_seed(args.manualSeed)


def flatten(list_of_lists):
    return itertools.chain.from_iterable(list_of_lists)

def train(model, train_loaders, criterion, optimizer, epoch, log):
    """Train for one epoch on the training set"""
    batch_time = AverageMeter()

    nat_losses = AverageMeter()
    nat_top1 = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()
    batch_idx = 0
    train_loaders = [iter(x) for x in train_loaders]
    len_dataloader = 0
    for x in train_loaders:
        len_dataloader += len(x)
    while True:
        for loader in train_loaders:
            input, target, _ = next(loader, (None, None, None))
            if input is None:
                return
            input = input.cuda()
            target = target.cuda()

            _, nat_output = model(input)
            
            nat_loss = criterion(nat_output, target)

            # measure accuracy and record loss
            nat_prec1 = accuracy(nat_output.data, target, topk=(1,))[0]
            nat_losses.update(nat_loss.data, input.size(0))
            nat_top1.update(nat_prec1, input.size(0))

            # compute gradient and do SGD step
            loss = nat_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if batch_idx % 10 == 0:
                log.debug('Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                        epoch, batch_idx, len_dataloader, batch_time=batch_time,
                        loss=nat_losses, top1=nat_top1))
            batch_idx += 1

def validate(val_loader, model, criterion, epoch, log, method):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (input, target, _) in enumerate(val_loader):
            input = input.cuda()
            target = target.cuda()
            # compute output
            _, output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target, topk=(1,))[0]
            losses.update(loss.data, input.size(0))
            top1.update(prec1, input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                log.debug('Validate: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                        i, len(val_loader), batch_time=batch_time, loss=losses,
                        top1=top1))

    log.debug(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
    return top1.avg

def adjust_learning_rate(args, optimizer, epoch):
    lr = args.lr
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def main():

    log = logging.getLogger(__name__)
    formatter = logging.Formatter('%(asctime)s : %(message)s')
    fileHandler = logging.FileHandler(os.path.join(directory, args.log_name), mode='w')
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)
    log.setLevel(logging.DEBUG)
    log.addHandler(fileHandler)
    log.addHandler(streamHandler) 

    if args.in_dataset == "color_mnist":
        train_loader1 = get_biased_mnist_dataloader(args, root = './datasets/MNIST', batch_size=args.batch_size,
                                        data_label_correlation= args.data_label_correlation,
                                        n_confusing_labels= args.num_classes - 1,
                                        train=True, partial=True, cmap = "1")
        train_loader2 = get_biased_mnist_dataloader(args, root = './datasets/MNIST', batch_size=args.batch_size,
                                        data_label_correlation= args.data_label_correlation,
                                        n_confusing_labels= args.num_classes - 1,
                                        train=True, partial=True, cmap = "2")
        val_loader = get_biased_mnist_dataloader(args, root = './datasets/MNIST', batch_size=args.batch_size,
                                        data_label_correlation= args.data_label_correlation,
                                        n_confusing_labels= args.num_classes - 1,
                                        train=False, partial=True, cmap = "1")
    elif args.in_dataset == "waterbird":
        train_loader = get_waterbird_dataloader(args, data_label_correlation=args.data_label_correlation, split="train")
        val_loader = get_waterbird_dataloader(args, data_label_correlation=args.data_label_correlation, split="val")
    elif args.in_dataset == "celebA":
        train_loader = get_celebA_dataloader(args, split="train")
        val_loader = get_celebA_dataloader(args, split="val")

    if args.model_arch == 'resnet18':
        pretrained = True
        if args.in_dataset == 'color_mnist':
            pretrained = False #True for celebA & waterbird ; False for Color_MNIST
        base_model = load_model(pretrained) 
    if torch.cuda.device_count() > 1:
        base_model = torch.nn.DataParallel(base_model)


    if args.method == "erm":
        model = base_model.cuda()
        criterion = nn.CrossEntropyLoss().cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        assert False, 'Not supported method: {}'.format(args.method)
    
    cudnn.benchmark = True

    freeze_bn_affine = False
    def freeze_bn(model, freeze_bn_affine=True):
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                if freeze_bn_affine:
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False
    freeze_bn(model, freeze_bn_affine)
    
    if args.in_dataset == "color_mnist":
        train_loaders = [train_loader1, train_loader2]
    elif args.in_dataset == "waterbird" or args.in_dataset == "celebA":
        train_loaders = [train_loader]

    for epoch in range(args.start_epoch, args.epochs):
        print(f"Start training epoch {epoch}")
        adjust_learning_rate(args, optimizer, epoch)
        train(model, train_loaders, criterion, optimizer, epoch, log) 
        prec1 = validate(val_loader, model, criterion, epoch, log, args.method)
        if (epoch + 1) % args.save_epoch == 0:
            save_checkpoint(args, {
                    'epoch': epoch + 1,
                    'state_dict_model': model.state_dict(),
            }, epoch + 1) 
if __name__ == '__main__':
    main()
