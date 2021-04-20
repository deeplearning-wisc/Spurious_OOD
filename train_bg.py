import argparse
import os
import sys
import shutil
import time
import random
import json
import logging
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
import utils.svhn_loader as svhn
from models.fine_tuning_layer import clssimp as clssimp
import models.densenet as dn
import models.wideresnet as wn
import models.resnet as rn
import models.simplenet as sn
from utils import LinfPGDAttack, TinyImages
# used for logging to TensorBoard
from tensorboard_logger import configure, log_value
from torch.distributions.multivariate_normal import MultivariateNormal

from torch.utils.data import Sampler
from utils.pascal_voc_loader import pascalVOCSet
from utils import cocoloader
from utils.transform import ReLabel, ToLabel, ToSP, Scale
from utils import ImageNet

from datasets.color_mnist import get_biased_mnist_dataloader
from torch.autograd import grad
import models.simpleCNN as scnn

parser = argparse.ArgumentParser(description='OOD training for multi-label classification')

parser.add_argument('--in-dataset', default="color_mnist", type=str, help='in-distribution dataset e.g. IN-9')
parser.add_argument('--model-arch', default='resnet18', type=str, help='model architecture e.g. resnet101')
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
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    help='initial learning rate')
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
parser.add_argument('--data_label_correlation', default= 0.4, type=float,
                    help='data_label_correlation')
# saving, naming and logging
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--name', default = "irm_test_0.4_3rd", type=str,
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

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    img_transform = transforms.Compose([
                                        # transforms.RandomResizedCrop((256),scale=(0.5, 2.0)),
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        normalize])
    val_transform = transforms.Compose([transforms.Scale(256),
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
        # train_set = torchvision.datasets.ImageFolder(root="datasets/random_shape/train_4", transform=img_transform)
        # val_set = torchvision.datasets.ImageFolder(root="datasets/random_shape/val_4", transform=val_transform)    
        num_classes = 9
        lr_schedule=[50, 75, 90]
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, num_workers= 4, shuffle=True, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, num_workers= 4, shuffle=False, pin_memory=True)
    elif args.in_dataset == "color_mnist":
            train_loader1 = get_biased_mnist_dataloader(root = './datasets/MNIST', batch_size=args.batch_size,
                                            data_label_correlation= args.data_label_correlation,
                                            n_confusing_labels= 4,
                                            train=True, partial=True, cmap = "1")
            train_loader2 = get_biased_mnist_dataloader(root = './datasets/MNIST', batch_size=args.batch_size,
                                            data_label_correlation= args.data_label_correlation,
                                            n_confusing_labels= 4,
                                            train=True, partial=True, cmap = "2")
            val_loader = get_biased_mnist_dataloader(root = './datasets/MNIST', batch_size=args.batch_size,
                                            data_label_correlation= args.data_label_correlation,
                                            n_confusing_labels= 4,
                                            train=False, partial=True, cmap = "1")
            num_classes = 5
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
    elif args.model_arch == "resnet101":
        # orig_resnet = torchvision.models.resnet101(pretrained=True)
        orig_resnet = rn.l_resnet101()
        rn_checkpoint = torch.load("R-101-GN-WS.pth.tar")
        from collections import OrderedDict
        new_checkpoint = OrderedDict([(k[7:], v) for k, v in rn_checkpoint.items()])
        orig_resnet.load_state_dict(new_checkpoint)
        features = list(orig_resnet.children())
        model = nn.Sequential(*features[0:8])
        clsfier = clssimp(2048, num_classes )
    else:
        assert False, 'Not supported model arch: {}'.format(args.model_arch)


    model = model.cuda()
    clsfier = clsfier.cuda()

    cudnn.benchmark = True

    # criterion = nn.BCEWithLogitsLoss().cuda() # sigmoid 
    # weights = [8.0, 1.0, 1.0, 1.0, 1.0]
    # class_weights = torch.FloatTensor(weights)
    # criterion = nn.CrossEntropyLoss(weight=class_weights).cuda()


    criterion = nn.CrossEntropyLoss().cuda()
    # optimizer = torch.optim.Adam([{'params': model.parameters(),'lr':args.lr/10},
    #                             {'params': clsfier.parameters()}], lr=args.lr)



    # model = scnn.ConvNet().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # optimizer = torch.optim.SGD([{'params': model.parameters(),'lr':args.lr/10},
    #                             {'params': clsfier.parameters()}], args.lr,
    #                             momentum=args.momentum,
    #                             nesterov=True,
    #                             weight_decay=args.weight_decay)

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
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
            if freeze_bn_affine:
                m.weight.requires_grad = False
                m.bias.requires_grad = False
    #CORE

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, lr_schedule)

        # train for one epoch
        # train(train_loader, model, clsfier, criterion, optimizer, epoch, log)
        # train_contrast(train_loader1, train_loader2, model, clsfier, criterion, optimizer, epoch, log)
        irm_train(model, clsfier, [train_loader1, train_loader2], criterion, optimizer, epoch)
        # evaluate on validation set

        prec1 = validate(val_loader, model, clsfier, criterion, epoch, log)

        # remember best prec@1 and save checkpoint
        if (epoch + 1) % args.save_epoch == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict_model': model.state_dict(),
                'state_dict_clsfier': clsfier.state_dict(),
            }, epoch + 1) 

def train(train_loader, model, clsfier, criterion, optimizer, epoch, log):
    """Train for one epoch on the training set"""
    batch_time = AverageMeter()

    nat_losses = AverageMeter()
    nat_top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target, _) in enumerate(train_loader):
        input = input.cuda()
        target = target.cuda()

        nat_output = model(input)
        nat_output = clsfier(nat_output)
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

        if i % args.print_freq == 0:
            log.debug('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      loss=nat_losses, top1=nat_top1))

    # log to TensorBoard
    if args.tensorboard:
        log_value('nat_train_loss', nat_losses.avg, epoch)
        log_value('nat_train_acc', nat_top1.avg, epoch)

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
  g1 = grad(losses[0::2].mean(), dummy, create_graph=True)[0]
  g2 = grad(losses[1::2].mean(), dummy, create_graph=True)[0]
  return (g1 * g2).sum()

def irm_train(model, clsfier, train_loaders, criterion, optimizer, epoch):
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
      output = clsfier(model(data))
      one_hot_target = torch.nn.functional.one_hot(target).float()
      loss_erm = F.binary_cross_entropy_with_logits(output * dummy_w, one_hot_target, reduction='none')
      # loss_erm = criterion(output, target)
      penalty += compute_irm_penalty(loss_erm, dummy_w)
      error += loss_erm.mean()
    (error + penalty_multiplier * penalty).backward()
    optimizer.step()
    if batch_idx % 10 == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tERM loss: {:.6f}\tGrad penalty: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loaders[0]),
               100. * batch_idx / len(train_loaders[0]), error.item(), penalty.item()))
      # print('First 20 logits', output.data.cpu().numpy()[:20])

    batch_idx += 1

def rebias_train(f_model, g_model, train_loaders, f_optimizer, g_optimizer, epoch):
    outer_criterion_config={'sigma_x': 1, 'sigma_y': 1, 'algorithm': 'unbiased'},
    inner_criterion_config={'sigma_x': 1, 'sigma_y': 1, 'algorithm': 'unbiased'},
    inner_criterion = MinusRbfHSIC(**inner_criterion_config)
    outer_criterion = RbfHSIC(**outer_criterion_config)
    classification_criterion = nn.CrossEntropyLoss()
    
    def update_g(model, data, target, g_lambda_inner=1):
        model.train()
        
        g_loss = 0
        for g_idx, g_net in enumerate(model.g_nets):
            preds, g_feats = g_net(data)
            _g_loss = 0

            _g_loss_cls = classification_criterion(preds, target)
            _g_loss += _g_loss_cls

            _, f_feats = model.f_net(data)
            _g_loss_inner = inner_criterion(g_feats, f_feats, labels=target)
            _g_loss += g_lambda_inner * _g_loss_inner
        
        g_loss += _g_loss
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

    def update_f(model, data, target, f_lambda_outer=1):
        model.train()

        f_loss = 0
        preds, f_feats = model.f_net(data)
        f_loss_cls = classification_criterion(preds, target)
        f_loss += f_loss_cls

        f_loss_indep = 0
        for g_idx, g_net in enumerate(model.g_nets):
            _g_preds, _g_feats = g_net(x)
            _f_loss_indep = outer_criterion(f_feats, _g_feats, labels=target, f_pred=preds)
            f_loss_indep += _f_loss_indep
        f_loss += f_lambda_outer * f_loss_indep

        f_optimizer.zero_grad()
        f_loss.backward()
        f_optimizer.step()

    model = ReBiasModels(f_model, g_model)
    for i in range(epoch):
        for loader in train_loaders:
            model.train()
            data, target, _ = next(loader, (None, None, None))
            if data is None:
                return
            data, target = data.cuda(), target.cuda()
            for _ in range(n_g_update):
                update_g(model, data, target)
            update_f(model, data, target)

def validate(val_loader, model, clsfier, criterion, epoch, log):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    clsfier.eval()
    with torch.no_grad():
        end = time.time()
        for i, (input, target, _) in enumerate(val_loader):
            input = input.cuda()
            target = target.cuda()
            # compute output
            output = model(input)
            output = clsfier(output)
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