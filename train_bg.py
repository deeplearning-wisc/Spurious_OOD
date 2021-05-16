import argparse
import os
import sys
import shutil
import time
import random
import json
import logging
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision
from models.fine_tuning_layer import clssimp as clssimp
import models.densenet as dn
import models.wideresnet as wn
import models.resnet as rn
import models.simplenet as sn
from models import CNNModel, res50
# used for logging to TensorBoard
from tensorboard_logger import configure, log_value
# from torch.distributions.multivariate_normal import MultivariateNormal

from torch.utils.data import Sampler, DataLoader
from rebias_utils import SimpleConvNet, RbfHSIC, MinusRbfHSIC, ReBiasModels

from datasets.color_mnist import get_biased_mnist_dataloader
from datasets.cub_dataset import WaterbirdDataset
from torch.autograd import grad
import models.simpleCNN as scnn

parser = argparse.ArgumentParser(description='OOD training for multi-label classification')

parser.add_argument('--in-dataset', default="color_mnist", type=str, help='in-distribution dataset e.g. IN-9')
parser.add_argument('--model-arch', default='general_model', type=str, help='model architecture e.g. resnet101')
parser.add_argument('--method', default='erm', type=str, help='method used for model training')
parser.add_argument('--save-epoch', default= 10, type=int,
                    help='save the model every save_epoch, default = 10') # freq; save model state_dict()
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    help='print frequency (default: 10)') # print every print-freq batches during training
# ID train & val batch size
parser.add_argument('-b', '--batch-size', default= 64, type=int,
                    help='mini-batch size (default: 64) used for training')
# training schedule
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--epochs', default= 10, type=int,
                    help='number of total epochs to run, default = 30')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--num-classes', default=2, type=int,
                    help='number of classes for model training')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', '--wd', default=0.0001, type=float,
                    help='weight decay (default: 0.0001)')
# # densenet
# parser.add_argument('--layers', default= 100, type=int,
#                     help='total number of layers (default: 100) for DenseNet')
# parser.add_argument('--growth', default= 12, type=int,
#                     help='number of new channels per layer (default: 12)')
# ## network spec
# parser.add_argument('--droprate', default=0.0, type=float,
#                     help='dropout probability (default: 0.0)')
# parser.add_argument('--no-augment', dest='augment', action='store_false',
#                     help='whether to use standard augmentation (default: True)')
# parser.add_argument('--reduce', default=0.5, type=float,
#                     help='compression rate in transition stage (default: 0.5)')
# parser.add_argument('--no-bottleneck', dest='bottleneck', action='store_false',
#                     help='To not use bottleneck block')
parser.add_argument('--data_label_correlation', default= 1, type=float,
                    help='data_label_correlation')
# saving, naming and logging
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--name', default = "erm_test_1_debug", type=str,
                    help='name of experiment')
parser.add_argument('--tensorboard',
                    help='Log progress to TensorBoard', action='store_true')
parser.add_argument('--log_name',
                    help='Name of the Log File', type = str, default = "info.log")
#Device options
parser.add_argument('--gpu-ids', default='5', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')

parser.set_defaults(bottleneck=True)
parser.set_defaults(augment=True)

args = parser.parse_args()

state = {k: v for k, v in args._get_kwargs()}
print(state)
directory = "checkpoints/{in_dataset}/{name}/".format(in_dataset=args.in_dataset, name=args.name)
if not os.path.exists(directory):
    os.makedirs(directory)
save_state_file = os.path.join(directory, 'args.txt')
fw = open(save_state_file, 'w')
print(state, file=fw)
fw.close()

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
use_cuda = torch.cuda.is_available()
devices = list(range(torch.cuda.device_count()))

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
torch.manual_seed(args.manualSeed)
np.random.seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

def flatten(list_of_lists):
    return itertools.chain.from_iterable(list_of_lists)

def main():
    if args.tensorboard: configure("runs/%s"%(args.name))

    log = logging.getLogger(__name__)
    formatter = logging.Formatter('%(asctime)s : %(message)s')
    fileHandler = logging.FileHandler(os.path.join(directory, args.log_name), mode='w')
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)
    log.setLevel(logging.DEBUG)
    log.addHandler(fileHandler)
    log.addHandler(streamHandler) 

    # Image trannsform for natural datasets (**Not applicable for ColorMNIST)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    img_transform = transforms.Compose([
                                        # transforms.RandomResizedCrop((256),scale=(0.5, 2.0)),
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        normalize])
    val_transform = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        normalize])
    # label_transform = transforms.Compose([ToLabel()])

    if args.in_dataset == "IN-9":
        train_set = torchvision.datasets.ImageFolder(root="/nobackup-slow/dataset/background/original/train", transform=img_transform)
        val_set = torchvision.datasets.ImageFolder(root="/nobackup-slow/dataset/background/original/val", transform=val_transform)    
        num_classes = 9
        lr_schedule=[50, 75, 90]
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, num_workers= 4, shuffle=True, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, num_workers= 4, shuffle=False, pin_memory=True)
    elif args.in_dataset == "random":
        train_set = torchvision.datasets.ImageFolder(root="/nobackup-slow/dataset/shape/train", transform=img_transform)
        val_set = torchvision.datasets.ImageFolder(root="/nobackup-slow/dataset/shape/val", transform=val_transform)    
        num_classes = 9
        lr_schedule=[50, 75, 90]
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, num_workers= 4, shuffle=True, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, num_workers= 4, shuffle=False, pin_memory=True)
    elif args.in_dataset == "color_mnist":
            train_loader1 = get_biased_mnist_dataloader(root = './datasets/MNIST', batch_size=args.batch_size,
                                            data_label_correlation= args.data_label_correlation,
                                            n_confusing_labels= args.num_classes - 1,
                                            train=True, partial=True, cmap = "1")
            train_loader2 = get_biased_mnist_dataloader(root = './datasets/MNIST', batch_size=args.batch_size,
                                            data_label_correlation= args.data_label_correlation,
                                            n_confusing_labels= args.num_classes - 1,
                                            train=True, partial=True, cmap = "2")
            val_loader = get_biased_mnist_dataloader(root = './datasets/MNIST', batch_size=args.batch_size,
                                            data_label_correlation= args.data_label_correlation,
                                            n_confusing_labels= args.num_classes - 1,
                                            train=False, partial=True, cmap = "1")
            lr_schedule=[50, 75, 90]
    elif args.in_dataset == "color_mnist_multi":
            train_loader1 = get_biased_mnist_dataloader(root = './datasets/MNIST', batch_size=args.batch_size,
                                            data_label_correlation= args.data_label_correlation,
                                            n_confusing_labels= args.num_classes - 1,
                                            train=True, partial=True, cmap = "1")
            train_loader2 = get_biased_mnist_dataloader(root = './datasets/MNIST', batch_size=args.batch_size,
                                            data_label_correlation= args.data_label_correlation,
                                            n_confusing_labels= args.num_classes - 1,
                                            train=True, partial=True, cmap = "2")
            train_loader3 = get_biased_mnist_dataloader(root = './datasets/MNIST', batch_size=args.batch_size,
                                            data_label_correlation= args.data_label_correlation,
                                            n_confusing_labels= args.num_classes - 1,
                                            train=True, partial=True, cmap = "3")
            train_loader4 = get_biased_mnist_dataloader(root = './datasets/MNIST', batch_size=args.batch_size,
                                            data_label_correlation= args.data_label_correlation,
                                            n_confusing_labels= args.num_classes - 1,
                                            train=True, partial=True, cmap = "4")
            val_loader = get_biased_mnist_dataloader(root = './datasets/MNIST', batch_size=args.batch_size,
                                            data_label_correlation= args.data_label_correlation,
                                            n_confusing_labels= args.num_classes - 1,
                                            train=False, partial=True, cmap = "1")
            lr_schedule=[50, 75, 90]
    elif args.in_dataset == "waterbird":
        train_dataset = WaterbirdDataset(data_correlation=0.95, train=True)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_dataset = WaterbirdDataset(data_correlation=0.95, train=False)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
        lr_schedule=[50, 75, 90]

    # create model
    if args.model_arch == 'densenet':
        # model = dn.DenseNet3(args.layers, num_classes, args.growth, reduction=args.reduce,
        #                      bottleneck=args.bottleneck, dropRate=args.droprate, normalizer=normalizer)
        orig_densenet = torchvision.models.densenet121(pretrained=True)
        features = list(orig_densenet.features)
        model = nn.Sequential(*features, nn.ReLU(inplace=True))
        clsfier = clssimp(1024, num_classes)
        # model = torchvision.models.densenet121(pretrained=False)
    elif args.model_arch == "wideresnet50":
        orig_resnet = torchvision.models.wide_resnet50_2(pretrained=True)
        features = list(orig_resnet.children())
        model = nn.Sequential(*features[0:8])
        clsfier = clssimp(2048, num_classes)
    elif args.model_arch == "resnet18":
        orig_resnet = torchvision.models.resnet18(pretrained=True)
        features = list(orig_resnet.children())
        model = nn.Sequential(*features[0:8])
        clsfier = clssimp(512, num_classes)
    # elif args.model_arch == "resnet101":
    #     # orig_resnet = torchvision.models.resnet101(pretrained=True)
    #     orig_resnet = rn.l_resnet101()
    #     rn_checkpoint = torch.load("R-101-GN-WS.pth.tar")
    #     from collections import OrderedDict
    #     new_checkpoint = OrderedDict([(k[7:], v) for k, v in rn_checkpoint.items()])
    #     orig_resnet.load_state_dict(new_checkpoint)
    #     features = list(orig_resnet.children())
    #     model = nn.Sequential(*features[0:8])
    #     clsfier = clssimp(2048, num_classes )
    elif args.model_arch == "general_model":
        base_model = CNNModel(num_classes=args.num_classes, bn_init=True, method=args.method)
    elif args.model_arch == "resnet50":
        base_model = res50(n_classes=args.num_classes, method=args.method)
    else:
        assert False, 'Not supported model arch: {}'.format(args.model_arch)

    # Method declaration
    if args.method == "dann" or args.method == "cdann" or args.method == "erm" \
                    or args.method == "irm" or args.method == "rex"  or args.method == "gdro" or args.method == "mixup":
        model = base_model.cuda()
    elif args.method == "rebias":
        n_g_nets = 1
        f_model = base_model.cuda()
        g_model = [base_model.cuda() for _ in range(n_g_nets)]
    else:
        assert False, 'Not supported method: {}'.format(args.method)
    
    cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss().cuda()
    
    if args.method == "rebias":
        f_optimizer = torch.optim.Adam(f_model.parameters(), lr=args.lr)
        g_optimizer = torch.optim.Adam(flatten([g_net.parameters() for g_net in g_model]), lr=args.lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            # checkpoint = torch.load(args.resume, map_location = 'cuda:0') # (if not Dataparallel) loads the model to a given GPU device
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict_model'])
            clsfier.load_state_dict(checkpoint['state_dict_clsfier'])
            # model = model.to('cuda:0') # (if not Dataparallel) convert the model’s parameter tensors to CUDA tensors
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    freeze_bn_affine = True
    def freeze_bn(model, freeze_bn_affine=True):
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                if freeze_bn_affine:
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False
    
    if args.method == "rebias":
        freeze_bn(f_model, freeze_bn_affine)
        for g_net in g_model:
            freeze_bn(g_net, freeze_bn_affine)
    else:
        freeze_bn(model, freeze_bn_affine)
    
    #CORE
    if args.in_dataset == "color_mnist_multi":
        train_loaders = [train_loader1, train_loader2, train_loader3, train_loader4]
    elif args.in_dataset == "color_mnist":
        train_loaders = [train_loader1, train_loader2]
    elif args.in_dataset == "waterbird":
        train_loaders = [train_loader]

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        # train(train_loader, model, clsfier, criterion, optimizer, epoch, log)
        # train_contrast(train_loader1, train_loader2, model, clsfier, criterion, optimizer, epoch, log)
        if args.method == "rebias":
            rebias_train(f_model, g_model, train_loaders, f_optimizer, g_optimizer, epoch)
            prec1 = rebias_validate(f_model, val_loader, epoch, log)
            if (epoch + 1) % args.save_epoch == 0:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict_model': f_model.state_dict(),
                }, epoch + 1) 
        elif args.method == "dann" or args.method == "cdann":
            dann_train(model, train_loaders, optimizer, epoch, args.epochs, cdann=(args.method == "cdann"))
            prec1 = dann_validate(model, val_loader, epoch, log, cdann=(args.method == "cdann"))
            if (epoch + 1) % args.save_epoch == 0:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict_model': model.state_dict(),
                }, epoch + 1)       
        elif args.method == "mixup":
            mixup_alpha = 1
            mixup_train(model, optimizer, train_loader1, train_loader2, criterion, mixup_alpha, epoch, log)
            prec1 = validate(val_loader, model, criterion, epoch, log)
            if (epoch + 1) % args.save_epoch == 0:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict_model': model.state_dict(),
                }, epoch + 1)               
        elif args.method == "irm":
            adjust_learning_rate(optimizer, epoch, lr_schedule)
            irm_train_v2(model, train_loaders, criterion, optimizer, epoch)  

            prec1 = validate(val_loader, model, criterion, epoch, log)

            if (epoch + 1) % args.save_epoch == 0:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict_model': model.state_dict(),
                }, epoch + 1)        
        elif args.method == "rex":
            adjust_learning_rate(optimizer, epoch, lr_schedule)
            rex_train(model, train_loaders, criterion, optimizer, epoch)  
            prec1 = validate(val_loader, model, criterion, epoch, log)
            if (epoch + 1) % args.save_epoch == 0:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict_model': model.state_dict(),
                }, epoch + 1)    
        elif args.method == "gdro":
            adjust_learning_rate(optimizer, epoch, lr_schedule)
            gdro_train(model, train_loaders, criterion, optimizer, epoch)  
            prec1 = validate(val_loader, model, criterion, epoch, log)
            if (epoch + 1) % args.save_epoch == 0:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict_model': model.state_dict(),
                }, epoch + 1)      
        elif args.method == "erm":
            adjust_learning_rate(optimizer, epoch, lr_schedule)
            train(model, train_loaders, criterion, optimizer, epoch, log)  
            prec1 = validate(val_loader, model, criterion, epoch, log)
            if (epoch + 1) % args.save_epoch == 0:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict_model': model.state_dict(),
                }, epoch + 1) 
    

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
                        epoch, batch_idx, len(train_loaders[0]) + len(train_loaders[1]), batch_time=batch_time,
                        loss=nat_losses, top1=nat_top1))
            batch_idx += 1

    # log to TensorBoard
    if args.tensorboard:
        log_value('nat_train_loss', nat_losses.avg, epoch)
        log_value('nat_train_acc', nat_top1.avg, epoch)


def mixup_train(model, optimizer, train_loader1, train_loader2, criterion, mixup_alpha, epoch, log):
    model.train()
    lam = np.random.beta(mixup_alpha, mixup_alpha)

    for i, (set1, set2) in enumerate(zip(train_loader1, train_loader2)):
        optimizer.zero_grad()
        input1, target1, _ = set1
        input2, target2, _ = set2
        input1 = input1.cuda()
        target1 = target1.cuda()
        input2 = input2.cuda()
        target2 = target2.cuda()
        input = lam * input1 + (1 - lam) * input2

        _, pred = model(input)
        loss_1 = lam * criterion(pred, target1)
        loss_2 = (1 - lam) * criterion(pred, target2)
        loss = loss_1 + loss_2
        loss.backward()
        optimizer.step()


def train_contrast(train_loader1, train_loader2, model, clsfier, criterion, optimizer, epoch, log):
    """Train for one epoch on the training set"""
    batch_time = AverageMeter()

    nat_losses = AverageMeter()
    nat_top1 = AverageMeter()
    
    def avg_energy(outputs, targets, bias = 0):
        zero_outputs = outputs[torch.nonzero(targets == bias).squeeze()]
        return -torch.logsumexp(zero_outputs, dim=1).mean()
    # switch to train mode
    model.train()

    end = time.time()
    for i, (set1, set2) in enumerate(zip(train_loader1, train_loader2)):
        input1, target1, _ = set1
        input2, target2, _ = set2
        input1 = input1.cuda()
        target1 = target1.cuda()
        input2 = input2.cuda()
        target2 = target2.cuda()
        
        output1 = clsfier(model(input1))
        output2 = clsfier(model(input2))
        energy_loss_0 = torch.abs(avg_energy(output1, target1, 0) - avg_energy(output2, target2, 0))
        energy_loss_1 = torch.abs(avg_energy(output1, target1, 1) - avg_energy(output2, target2, 1))
        output = torch.cat((output1, output2), 0)
        target = torch.cat((target1, target2), 0)
        nat_loss = criterion(output, target)

        # measure accuracy and record loss
        nat_prec1 = accuracy(output.data, target, topk=(1,))[0]
        nat_losses.update(nat_loss.data, output.size(0))
        nat_top1.update(nat_prec1, output.size(0))

        # compute gradient and do SGD step
        loss = nat_loss + 0 * energy_loss_0 + + 0 * energy_loss_1


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            log.debug('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader1), batch_time=batch_time,
                      loss=nat_losses, top1=nat_top1))

    # log to TensorBoard
    if args.tensorboard:
        log_value('nat_train_loss', nat_losses.avg, epoch)
        log_value('nat_train_acc', nat_top1.avg, epoch)

def compute_irm_penalty(losses, dummy):
  g1 = grad(losses[0::2].mean(), dummy, create_graph=True)[0] # dim 1 e.g. tensor([0.2479])
  g2 = grad(losses[1::2].mean(), dummy, create_graph=True)[0]
  return (g1 * g2).sum() # dim 0 e.g. tensor(0.0439)

def irm_train(model, train_loaders, criterion, optimizer, epoch):
  model.train()
  
  train_loaders = [iter(x) for x in train_loaders]
  dummy_w = torch.nn.Parameter(torch.Tensor([1.0])).cuda()
  batch_idx = 0
  penalty_multiplier = epoch ** 1.1
  print(f'Using penalty multiplier {penalty_multiplier}')
  while True:
        optimizer.zero_grad()
        error = 0
        penalty = 0
        for loader in train_loaders:
            data, target, _ = next(loader, (None, None, None))
            if data is None:
                return
            data, target = data.cuda(), target.cuda()
            _, output = model(data)
            one_hot_target = torch.nn.functional.one_hot(target).float()
            loss_erm = F.binary_cross_entropy_with_logits(output * dummy_w, one_hot_target, reduction='none')
            penalty += compute_irm_penalty(loss_erm, dummy_w)
            error += loss_erm.mean()
        (error + penalty_multiplier * penalty).backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tIRM loss: {:.6f}\tGrad penalty: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loaders[0]),
                    100. * batch_idx / len(train_loaders[0]), error.item(), penalty.item()))
        # print('First 20 logits', output.data.cpu().numpy()[:20])

        batch_idx += 1

def irm_train_v2(model, train_loaders, criterion, optimizer, epoch):
    '''
    F.cross_entropy()

    '''
    model.train()
    train_loaders = [iter(x) for x in train_loaders]
    dummy_w = torch.nn.Parameter(torch.Tensor([1.0])).cuda()
    batch_idx = 0
    penalty_multiplier = epoch ** 1.1
    print(f'Using penalty multiplier {penalty_multiplier}')
    while True:
        error = 0
        penalty = 0
        for loader in train_loaders:
            data, target, _ = next(loader, (None, None, None))
            if data is None:
                return
            data, target = data.cuda(), target.cuda()
            _, output = model(data)
            loss_erm = F.cross_entropy(output, target, reduction = "mean")
            loss_erm_for_penalty = F.cross_entropy(output * dummy_w, target, reduction = "none")
            penalty += compute_irm_penalty(loss_erm_for_penalty, dummy_w)
            error += loss_erm

        optimizer.zero_grad()
        error /= len(train_loaders)
        penalty /= len(train_loaders)
        (error  + penalty_multiplier * penalty).backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tIRM_V2 loss: {:.6f}\tGrad penalty: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loaders[0]),
                    100. * batch_idx / len(train_loaders[0]), error.item(), penalty.item()))
        # print('First 20 logits', output.data.cpu()numpy()[:20])

        batch_idx += 1

def rex_train(model, train_loaders, criterion, optimizer, epoch):
    '''
    REx adapted from DomainBed: https://github.com/facebookresearch/DomainBed/blob/master/domainbed/algorithms.py
    '''
    model.train()
    train_loaders = [iter(x) for x in train_loaders]
    # dummy_w = torch.nn.Parameter(torch.Tensor([1.0])).cuda()
    batch_idx = 0
    penalty_multiplier = 1.0
    # print(f'Using penalty multiplier {penalty_multiplier}')
    while True:
        losses = torch.zeros(len(train_loaders))
        for i, loader in enumerate(train_loaders):
            data, target, _ = next(loader, (None, None, None))
            if data is None:
                return
            data, target = data.cuda(), target.cuda()
            _, output = model(data)
            loss_erm = F.cross_entropy(output, target)
            losses[i] = loss_erm

        mean = losses.mean()
        penalty = ((losses - mean) ** 2).mean()
        optimizer.zero_grad()
        (mean + penalty_multiplier * penalty).backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tREx loss: {:.6f}\tGrad penalty: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loaders[0]),
                    100. * batch_idx / len(train_loaders[0]), mean.item(), penalty.item()))
        # print('First 20 logits', output.data.cpu()numpy()[:20])

        batch_idx += 1

def gdro_train(model, train_loaders, criterion, optimizer, epoch):
    '''
    GDRO (Robust ERM) minimizes the error of the worst group/domain/env
    Algorithm 1 from [https://arxiv.org/pdf/1911.08731.pdf]
    adapted from DomainBed
    '''
    groupdro_eta = 1e-2
    model.train()
    train_loaders = [iter(x) for x in train_loaders]
    batch_idx = 0
    penalty_multiplier = 1.0
    q = torch.ones(len(train_loaders)).cuda()
    while True:
        losses = torch.zeros(len(train_loaders)).cuda()
        for i, loader in enumerate(train_loaders):
            data, target, _ = next(loader, (None, None, None))
            if data is None:
                return
            data, target = data.cuda(), target.cuda()
            _, output = model(data)
            losses[i] = F.cross_entropy(output, target)
            q[i] = (groupdro_eta* losses[i].data).exp()

        q /= q.sum()
        loss = torch.dot(losses, q)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tGDRO loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loaders[0]),
                    100. * batch_idx / len(train_loaders[0]), loss.item()))
        batch_idx += 1

def rebias_train(f_model, g_model, train_loaders, f_optimizer, g_optimizer, epoch, n_g_update=1):
    outer_criterion_config={'sigma_x': 1, 'sigma_y': 1, 'algorithm': 'unbiased'}
    inner_criterion_config={'sigma_x': 1, 'sigma_y': 1, 'algorithm': 'unbiased'}
    inner_criterion = MinusRbfHSIC(**inner_criterion_config)
    outer_criterion = RbfHSIC(**outer_criterion_config)
    classification_criterion = nn.CrossEntropyLoss()
    
    def update_g(model, data, target, g_lambda_inner=1):
        model.train()
        
        g_loss = 0
        for g_idx, g_net in enumerate(model.g_nets):
            g_feats, preds = g_net(data)
            _g_loss = 0

            _g_loss_cls = classification_criterion(preds, target)
            _g_loss += _g_loss_cls

            f_feats, _ = model.f_net(data)
            _g_loss_inner = inner_criterion(g_feats, f_feats, labels=target)
            _g_loss += g_lambda_inner * _g_loss_inner
        
        g_loss += _g_loss
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

    def update_f(model, data, target, f_lambda_outer=1):
        model.train()

        f_loss = 0
        f_feats, preds = model.f_net(data)
        f_loss_cls = classification_criterion(preds, target)
        f_loss += f_loss_cls

        f_loss_indep = 0
        for g_idx, g_net in enumerate(model.g_nets):
            _g_feats, _g_preds = g_net(data)
            _f_loss_indep = outer_criterion(f_feats, _g_feats, labels=target, f_pred=preds)
            f_loss_indep += _f_loss_indep
        f_loss += f_lambda_outer * f_loss_indep

        f_optimizer.zero_grad()
        f_loss.backward()
        f_optimizer.step()

    model = ReBiasModels(f_model, g_model)
    train_loaders = [iter(x) for x in train_loaders]
    while True:
        for loader in train_loaders:
            model.train()
            data, target, _ = next(loader, (None, None, None))
            if data is None:
                return
            data, target = data.cuda(), target.cuda()
            for _ in range(n_g_update):
                update_g(model, data, target)
            update_f(model, data, target)

def rebias_validate(f_model, val_loader, epoch, log):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    classification_criterion = nn.CrossEntropyLoss()

    # switch to evaluate mode
    f_model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target, _) in enumerate(val_loader):
            input = input.cuda()
            target = target.cuda()
            # compute output
            _, output = f_model(input)
            loss = classification_criterion(output, target)

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target, topk=(1,))[0]
            losses.update(loss.data, input.size(0))
            top1.update(prec1, input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                log.debug('Test: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                        i, len(val_loader), batch_time=batch_time, loss=losses,
                        top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
    # log to TensorBoard
    if args.tensorboard:
        log_value('val_loss', losses.avg, epoch)
        log_value('val_acc', top1.avg, epoch)
    return top1.avg    

def dann_train_orig(model, source_train_loader, target_train_loader, optimizer, epoch, n_epoch, cdann=False):
    len_loader = min(len(source_train_loader), len(target_train_loader))
    source_train_loader = iter(source_train_loader)
    target_train_loader = iter(target_train_loader)
    loss_class = nn.CrossEntropyLoss()
    loss_domain = nn.CrossEntropyLoss()

    model.train()
    for i in range(len_loader):
        optimizer.zero_grad()
        p = float(i + epoch * len_loader) / n_epoch / len_loader
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        # Model training using source data
        src_img, src_label, _ = source_train_loader.next()
        src_domain_label = torch.zeros(len(src_label)).long().cuda()
        src_img, src_label = src_img.cuda(), src_label.cuda()

        if cdann:
            _, class_output, domain_output = model(input_data=src_img, alpha=alpha, y=src_label)
            y_counts = F.one_hot(src_label).sum(dim=0)
            weights = 1. / (y_counts[src_label] * y_counts.shape[0]).float()
            err_src_class = loss_class(class_output, src_label)
            err_src_domain = loss_domain(domain_output, src_domain_label)
            err_src_domain = (weights * err_src_domain).sum()
        else:
            _, class_output, domain_output = model(input_data=src_img, alpha=alpha)
            err_src_class = loss_class(class_output, src_label)
            err_src_domain = loss_domain(domain_output, src_domain_label)

        # Model training using target data
        tar_img, tar_label, _ = target_train_loader.next()
        tar_domain_label = torch.ones(len(tar_img)).long().cuda()
        tar_img, tar_label = tar_img.cuda(), tar_label.cuda()
        
        if cdann:
            _, class_output, domain_output = model(input_data=tar_img, alpha=alpha, y=tar_label)
            y_counts = F.one_hot(tar_label).sum(dim=0)
            weights = 1. / (y_counts[tar_label] * y_counts.shape[0]).float()
            err_tar_class = loss_class(class_output, tar_label)
            err_tar_domain = loss_domain(domain_output, tar_domain_label)
            err_tar_domain = (weights * err_tar_domain).sum()
        else:
            _, class_output, domain_output = model(input_data=tar_img, alpha=alpha)
            err_tar_class = loss_class(class_output, tar_label)
            err_tar_domain = loss_domain(domain_output, tar_domain_label)            

        err = err_src_class + err_src_domain + err_tar_class + err_tar_domain
        err.backward()
        optimizer.step()

def dann_train(model, train_loaders, optimizer, epoch, n_epoch, cdann=False):
    loss_class = nn.CrossEntropyLoss().cuda()
    loss_domain = nn.CrossEntropyLoss().cuda()
    len_loader = 0

    for x in train_loaders:
        len_loader += len(x)
    train_loaders = [iter(x) for x in train_loaders]

    while True:
        for (i, loader) in enumerate(train_loaders):
            optimizer.zero_grad()
            p = float(i + epoch * len_loader) / n_epoch / len_loader
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            model.train()
            data, target, _ = next(loader, (None, None, None))
            if data is None:
                return
            data, target = data.cuda(), target.cuda()
            domain_label = torch.full([len(data)], i).long().cuda()
            if cdann:
                _, class_output, domain_output = model(input_data=data, alpha=alpha, y=target)
                y_counts = F.one_hot(target).sum(dim=0)
                weights = 1. / (y_counts[target] * y_counts.shape[0]).float()
                err_src_class = loss_class(class_output, target)
                err_src_domain = loss_domain(domain_output, domain_label)
                err_src_domain = (weights * err_src_domain).sum()
            else:
                _, class_output, domain_output = model(input_data=data, alpha=alpha)
                err_src_class = loss_class(class_output, target)
                err_src_domain = loss_domain(domain_output, domain_label)

            err = err_src_class + err_src_domain
            err.backward()
            optimizer.step()

def dann_validate(model, val_loader, epoch, log, cdann=False):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    classification_criterion = nn.CrossEntropyLoss()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target, _) in enumerate(val_loader):
            input = input.cuda()
            target = target.cuda()
            # compute output
            if cdann:
                _, output = model(input, alpha=0, y=None)
            else:
                _, output, _ = model(input, alpha=0)
            loss = classification_criterion(output, target)

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target, topk=(1,))[0]
            losses.update(loss.data, input.size(0))
            top1.update(prec1, input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                log.debug('Test: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                        i, len(val_loader), batch_time=batch_time, loss=losses,
                        top1=top1))

    log.debug(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
    # log to TensorBoard
    if args.tensorboard:
        log_value('val_loss', losses.avg, epoch)
        log_value('val_acc', top1.avg, epoch)
    return top1.avg   

def validate(val_loader, model, criterion, epoch, log):
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
                log.debug('Test: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                        i, len(val_loader), batch_time=batch_time, loss=losses,
                        top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
    # log to TensorBoard
    if args.tensorboard:
        log_value('val_loss', losses.avg, epoch)
        log_value('val_acc', top1.avg, epoch)
    return top1.avg

def adjust_learning_rate(optimizer, epoch, lr_schedule=[50, 75, 90]):
    """Sets the learning rate to the initial LR decayed by 10 after 40 and 80 epochs"""
    lr = args.lr
    if epoch >= lr_schedule[0]:
        lr *= 0.1
    if epoch >= lr_schedule[1]:
        lr *= 0.1
    if epoch >= lr_schedule[2]:
        lr *= 0.1
    # lr = args.lr * (0.1 ** (epoch // 60)) * (0.1 ** (epoch // 80))
    # log to TensorBoard
    if args.tensorboard:
        log_value('learning_rate', lr, epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def save_checkpoint(state, epoch, name = None):
    """Saves checkpoint to disk"""
    directory = "checkpoints/{in_dataset}/{name}/".format(in_dataset=args.in_dataset, name=args.name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    if name == None:
        filename = directory + 'checkpoint_{}.pth.tar'.format(epoch)
    else: 
        filename = directory + '{}_{}.pth.tar'.format(name, epoch)
    torch.save(state, filename)

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

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__ == '__main__':
    main()
