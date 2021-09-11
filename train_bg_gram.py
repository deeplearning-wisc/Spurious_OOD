import argparse
import os
import sys
import shutil
import time
import random
import json
import math
import logging
import itertools
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from models import  bagnet18, SimpleConvNet, ReBiasModels

from torch.utils.data import Sampler, DataLoader
from utils import RbfHSIC, MinusRbfHSIC, AverageMeter, save_checkpoint, accuracy

from datasets.color_mnist import get_biased_mnist_dataloader
from datasets.cub_dataset import get_waterbird_dataloader
from datasets.celebA_dataset import get_celebA_dataloader
from torch.autograd import grad

parser = argparse.ArgumentParser(description='use resnet18 (not pretrained) as in the gram matrix method')

parser.add_argument('--in-dataset', default="waterbird", type=str, help='in-distribution dataset e.g. IN-9')
parser.add_argument('--model-arch', default='resnet18', type=str, help='model architecture e.g. resnet50')
parser.add_argument('--domain-num', default=4, type=int,
                    help='the number of environments for model training')
parser.add_argument('--method', default='erm', type=str, help='method used for model training')
parser.add_argument('--save-epoch', default=25, type=int,
                    help='save the model every save_epoch, default = 10') # freq; save model state_dict()
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    help='print frequency (default: 10)') # print every print-freq batches during training
# ID train & val batch size
parser.add_argument('-b', '--batch-size', default= 64, type=int,
                    help='mini-batch size (default: 64) used for training')
# training schedule
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--epochs', default=50, type=int,
                    help='number of total epochs to run, default = 30')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--num-classes', default=2, type=int,
                    help='number of classes for model training')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', '--wd', default=0.05, type=float,
                    help='weight decay (default: 0.0001)')
parser.add_argument('--data_label_correlation', default=0.7, type=float,
                    help='data_label_correlation')
# saving, naming and logging
parser.add_argument('--exp-name', default = 'erm_rebuttal_0.8', type=str, 
                    help='help identify checkpoint')
parser.add_argument('--name', default="erm_rebuttal", type=str,
                    help='name of experiment')
parser.add_argument('--log_name', type = str, default = "info.log",
                    help='Name of the Log File')
# Device options
parser.add_argument('--gpu-ids', default='1', type=str,
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
parser.add_argument('--lr_decay_epochs', type=str, default='10,15,20',
                        help='where to decay lr, can be a list')
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


import torchvision
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
# from utils.dann_utils import ReverseLayerF
        
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
    
    def forward(self, x):
        t = self.conv1(x)
        out = F.relu(self.bn1(t))
        t = self.conv2(out)
        out = self.bn2(self.conv2(out))   
        t = self.shortcut(x)
        out += t        
        out = F.relu(out)
        
        return out

    # workable version 
    def forward(self, x):
        t = self.conv1(x)
        out = F.relu(self.bn1(t))
        base_model.record(t)
        base_model.record(out)
        t = self.conv2(out)
        out = self.bn2(self.conv2(out))
        base_model.record(t)
        base_model.record(out)
        t = self.shortcut(x)
        out += t
        base_model.record(t)
        out = F.relu(out)
        base_model.record(out)
        
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=2):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = conv3x3(3,64)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.collecting = False
    
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        # out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        y = self.linear(out)
        return y
    
    def record(self, t):
        if self.collecting:
            self.gram_feats.append(t)
    
    def gram_feature_list(self,x):
        self.collecting = True
        self.gram_feats = []
        self.forward(x)
        self.collecting = False
        temp = self.gram_feats
        self.gram_feats = []
        return temp
    
    def load(self, path="resnet_svhn.pth"):
        tm = torch.load(path, map_location="cpu")        
        self.load_state_dict(tm)
    
    def get_min_max(self, data, power):
        mins = []
        maxs = []
        
        for i in range(0,len(data),128):
            batch = data[i:i+128].cuda()
            feat_list = self.gram_feature_list(batch)
            for L,feat_L in enumerate(feat_list):
                if L==len(mins):
                    mins.append([None]*len(power))
                    maxs.append([None]*len(power))
                
                for p,P in enumerate(power):
                    g_p = G_p(feat_L,P)
                    
                    current_min = g_p.min(dim=0,keepdim=True)[0]
                    current_max = g_p.max(dim=0,keepdim=True)[0]
                    
                    if mins[L][p] is None:
                        mins[L][p] = current_min
                        maxs[L][p] = current_max
                    else:
                        mins[L][p] = torch.min(current_min,mins[L][p])
                        maxs[L][p] = torch.max(current_max,maxs[L][p])
        
        return mins,maxs
    
    def get_deviations(self,data,power,mins,maxs):
        deviations = []
        
        for i in range(0,len(data),128):            
            batch = data[i:i+128].cuda()
            feat_list = self.gram_feature_list(batch)
            batch_deviations = []
            for L,feat_L in enumerate(feat_list):
                dev = 0
                for p,P in enumerate(power):
                    g_p = G_p(feat_L,P)
                    
                    dev +=  (F.relu(mins[L][p]-g_p)/torch.abs(mins[L][p]+10**-6)).sum(dim=1,keepdim=True)
                    dev +=  (F.relu(g_p-maxs[L][p])/torch.abs(maxs[L][p]+10**-6)).sum(dim=1,keepdim=True)
                batch_deviations.append(dev.cpu().detach().numpy())
            batch_deviations = np.concatenate(batch_deviations,axis=1)
            deviations.append(batch_deviations)
        deviations = np.concatenate(deviations,axis=0)
        
        return deviations

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

            # _, nat_output = model(input)
            nat_output = model(input)
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

def compute_irm_penalty(losses, dummy):
  g1 = grad(losses[0::2].mean(), dummy, create_graph=True)[0] # dim 1 e.g. tensor([0.2479])
  g2 = grad(losses[1::2].mean(), dummy, create_graph=True)[0]
  return (g1 * g2).sum() # dim 0 e.g. tensor(0.0439)

def irm_train(model, train_loaders, criterion, optimizer, epoch, log):
    '''
    F.cross_entropy()

    '''
    model.train()
    train_loaders = [iter(x) for x in train_loaders]
    dummy_w = torch.nn.Parameter(torch.Tensor([1.0])).cuda()
    batch_idx = 0
    penalty_multiplier = epoch ** args.penalty_multiplier
    log.debug(f'Using penalty multiplier {penalty_multiplier}')
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
            log.debug('Train Epoch: {} [{}/{} ({:.0f}%)]\tIRM loss: {:.6f}\tGrad penalty: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loaders[0]),
                    100. * batch_idx / len(train_loaders[0]), error.item(), penalty.item()))
        # print('First 20 logits', output.data.cpu()numpy()[:20])

        batch_idx += 1

def rex_train(model, train_loaders, criterion, optimizer, epoch, log):
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
            log.debug('Train Epoch: {} [{}/{} ({:.0f}%)]\tREx loss: {:.6f}\tGrad penalty: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loaders[0]),
                    100. * batch_idx / len(train_loaders[0]), mean.item(), penalty.item()))
        # print('First 20 logits', output.data.cpu()numpy()[:20])

        batch_idx += 1

def gdro_train(model, train_loaders, criterion, optimizer, epoch, log):
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
            log.debug('Train Epoch: {} [{}/{} ({:.0f}%)]\tGDRO loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loaders[0]),
                    100. * batch_idx / len(train_loaders[0]), loss.item()))
        batch_idx += 1

def rebias_train(f_model, g_model, train_loaders, f_optimizer, g_optimizer, epoch, log, n_g_update=1):
    '''
    Adapted from https://github.com/clovaai/rebias/blob/master/main_biased_mnist.py
    '''
    outer_criterion_config={'sigma_x': 1, 'sigma_y': 1, 'algorithm': 'unbiased'}
    inner_criterion_config={'sigma_x': 1, 'sigma_y': 1, 'algorithm': 'unbiased'}
    inner_criterion = MinusRbfHSIC(**inner_criterion_config)
    outer_criterion = RbfHSIC(**outer_criterion_config)
    classification_criterion = nn.CrossEntropyLoss()

    len_loader = 0
    for x in train_loaders:
        len_loader += len(x)
    train_loaders = [iter(x) for x in train_loaders]

    batch_time = AverageMeter()
    nat_losses = AverageMeter()
    nat_top1 = AverageMeter()
    batch_idx = 0

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

    def update_f(model, data, target, log, batch_idx, f_lambda_outer=1):
        end = time.time()
        model.train()
        f_optimizer.zero_grad()

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

        # measure accuracy and record loss
        nat_prec1 = accuracy(preds.data, target, topk=(1,))[0]
        nat_losses.update(f_loss.data, data.size(0))
        nat_top1.update(nat_prec1, data.size(0))        
        
        f_loss.backward()
        f_optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % 10 == 0:
            log.debug('Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    epoch, batch_idx, len_loader, batch_time=batch_time,
                    loss=nat_losses, top1=nat_top1))

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
            update_f(model, data, target, log, batch_idx)
            batch_idx += 1

def dann_train(model, train_loaders, optimizer, epoch, n_epoch, log, cdann=False):
    '''
    Adapted from DomainBed https://github.com/facebookresearch/DomainBed/blob/master/domainbed/algorithms.py
    '''
    loss_class = nn.CrossEntropyLoss().cuda()
    loss_domain = nn.CrossEntropyLoss().cuda()
    len_loader = 0

    for x in train_loaders:
        len_loader += len(x)
    train_loaders = [iter(x) for x in train_loaders]

    end = time.time()
    batch_time = AverageMeter()
    nat_losses = AverageMeter()
    nat_top1 = AverageMeter()
    batch_idx = 0

    while True:
        for (i, loader) in enumerate(train_loaders):
            optimizer.zero_grad()
            p = float(i + epoch * len_loader) / n_epoch / len_loader
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            model.train()
            data, target, env = next(loader, (None, None, None))
            if data is None:
                return
            data, target = data.cuda(), target.cuda()
            domain_label = env.long().cuda()
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

            # measure accuracy and record loss
            nat_prec1 = accuracy(class_output.data, target, topk=(1,))[0]
            nat_losses.update(err.data, data.size(0))
            nat_top1.update(nat_prec1, data.size(0))

            err.backward()
            optimizer.step()  

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if batch_idx % 10 == 0:
                log.debug('Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                        epoch, batch_idx, len_loader, batch_time=batch_time,
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
            if method == "dann":
                _, output, _ = model(input, alpha=0)
            elif method == "cdann":
                _, output = model(input, alpha=0, y=None)
            else:
              #   _, output = model(input)
                output = model(input)
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

if __name__ == '__main__':
   
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

    # base_model = Resnet(n_classes=args.num_classes, model=args.model_arch, method=args.method, domain_num=args.domain_num)
    base_model = ResNet(BasicBlock, [2,2,2,2], num_classes=2)
    if torch.cuda.device_count() > 1:
        base_model = torch.nn.DataParallel(base_model)


    if args.method in ["dann", "cdann", "erm", "irm", "rex", "gdro"]:
        model = base_model.cuda()
        criterion = nn.CrossEntropyLoss().cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.method == "rebias":
        f_model = base_model.cuda()
        criterion = nn.CrossEntropyLoss().cuda()
        if args.in_dataset == "waterbird" or args.in_dataset == "celebA":
            bnet = bagnet18(feature_pos='post', num_classes=2)
            g_model = [bnet.cuda() for _ in range(args.n_g_nets)]     
        elif args.in_dataset == "color_mnist":
            simnet = SimpleConvNet(num_classes=args.num_classes, kernel_size=1)
            g_model = [simnet.cuda() for _ in range(args.n_g_nets)]     
        else:
            assert False, 'Not supported g_model for dataset: {}'.format(args.in_dataset)   
        f_optimizer = torch.optim.Adam(f_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        g_optimizer = torch.optim.Adam(flatten([g_net.parameters() for g_net in g_model]), lr=args.lr, weight_decay=args.weight_decay)
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
    
    if args.method == "rebias":
        freeze_bn(f_model, freeze_bn_affine)
        for g_net in g_model:
            freeze_bn(g_net, freeze_bn_affine)
    else:
        freeze_bn(model, freeze_bn_affine)
    
    if args.in_dataset == "color_mnist":
        train_loaders = [train_loader1, train_loader2]
    elif args.in_dataset == "waterbird" or args.in_dataset == "celebA":
        train_loaders = [train_loader]

    for epoch in range(args.start_epoch, args.epochs):
        print(f"Start training epoch {epoch}")
        adjust_learning_rate(args, optimizer, epoch)

        if args.method == "rebias":
            rebias_train(f_model, g_model, train_loaders, f_optimizer, g_optimizer, epoch, log)
            model = f_model
        elif args.method == "dann" or args.method == "cdann":
            dann_train(model, train_loaders, optimizer, epoch, args.epochs, log, cdann=(args.method=="cdann"))    
        elif args.method == "irm":
            irm_train(model, train_loaders, criterion, optimizer, epoch, log)    
        elif args.method == "rex":
            rex_train(model, train_loaders, criterion, optimizer, epoch, log)  
        elif args.method == "gdro":
            gdro_train(model, train_loaders, criterion, optimizer, epoch, log)  
        elif args.method == "erm":
            train(model, train_loaders, criterion, optimizer, epoch, log) 
        
        prec1 = validate(val_loader, model, criterion, epoch, log, args.method)
        
        if (epoch + 1) % args.save_epoch == 0:
            save_checkpoint(args, {
                    'epoch': epoch + 1,
                    'state_dict_model': model.state_dict(),
            }, epoch + 1) 