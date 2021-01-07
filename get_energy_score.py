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
from matplotlib import pyplot as plt
from sklearn import metrics

import utils.svhn_loader as svhn
from models.fine_tuning_layer import clssimp as clssimp
import models.densenet as dn
import models.wideresnet as wn
import models.resnet as rn
import models.simplenet as sn
from neural_linear_mlc import NeuralLinear
from utils import LinfPGDAttack, TinyImages
# used for logging to TensorBoard
from tensorboard_logger import configure, log_value
from torch.distributions.multivariate_normal import MultivariateNormal

from torch.utils.data import Sampler
from utils.pascal_voc_loader import pascalVOCSet
from utils import cocoloader
from utils.transform import ReLabel, ToLabel, ToSP, Scale
from utils import ImageNet
from neural_linear_mlc import SimpleDataset

parser = argparse.ArgumentParser(description='OOD training for multi-label classification')

parser.add_argument('--in-dataset', default="pascal", type=str, help='in-distribution dataset e.g. pascal')
parser.add_argument('--auxiliary-dataset', default='imagenet', 
                    choices=['80m_tiny_images', 'imagenet'], type=str, help='which auxiliary dataset to use')
parser.add_argument('--model-arch', default='resnet101', type=str, help='model architecture e.g. simplenet densenet')

parser.add_argument('--save-epoch', default= 10, type=int,
                    help='save the model every save_epoch') # freq; save model state_dict()
parser.add_argument('--save-data-epoch', default= 120, type=int,
                    help='save the sampled ood every save_data_epoch') # freq; save model state_dict()
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    help='print frequency (default: 10)') # print every print-freq batches during training
# ID train & val batch size and OOD train batch size 
parser.add_argument('-b', '--batch-size', default= 32, type=int,
                    help='mini-batch size (default: 64) used for training id and ood')

parser.add_argument('--val-batch-size', default= 400, type=int,
                    help='mini-batch size (default: 400) used for ood validation')
# training schedule
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--epochs', default= 25, type=int,
                    help='number of total epochs to run')
parser.add_argument('--test_epoch', default= 24, type=int,
                    help='# epoch to test performance')

parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', '--wd', default=0.0001, type=float,
                    help='weight decay (default: 0.0001)')
# densenet
parser.add_argument('--layers', default= 100, type=int,
                    help='total number of layers (default: 100) for DenseNet')
parser.add_argument('--growth', default= 12, type=int,
                    help='number of new channels per layer (default: 12)')
## network spec
parser.add_argument('--droprate', default=0.0, type=float,
                    help='dropout probability (default: 0.0)')
parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='whether to use standard augmentation (default: True)')
parser.add_argument('--reduce', default=0.5, type=float,
                    help='compression rate in transition stage (default: 0.5)')
parser.add_argument('--no-bottleneck', dest='bottleneck', action='store_false',
                    help='To not use bottleneck block')
parser.add_argument('--beta', default=2.0, type=float, help='beta for out_loss')
parser.add_argument('--gamma', default=10.0, type=float, help='gamma for in_out_loss')
parser.add_argument('--eta', default=0.01, type=float, help='eta for energy loss')
parser.add_argument('--m_in', default=-15, type=float, help='M_in for energy loss')
parser.add_argument('--m_out', default=-3, type=float, help='M_out for energy loss')
# ood sampling and mining
parser.add_argument('--ood-batch-size', default= 200, type=int,
                    help='mini-batch size (default: 400) used for ood mining')
parser.add_argument('--pool-size', default= 100, type=int,
                    help='pool size')
parser.add_argument('--ood_factor', type=float, default= 1,
                 help='ood_dataset_size = len(train_loader.dataset) * ood_factor default = 2.0')
#posterior sampling
parser.add_argument('--a0', type=float, default=6.0, help='a0')
parser.add_argument('--b0', type=float, default=6.0, help='b0')
parser.add_argument('--lambda_prior', type=float, default=0.25, help='lambda_prior')
parser.add_argument('--sigma', type=float, default=10, help='control var for weights')
parser.add_argument('--sigma_n', type=float, default=0.5, help='control var for noise')
parser.add_argument('--conf', type=float, default=4.6, help='control ground truth for bayesian linear regression.3.9--0.98; 4.6 --0.99')
# saving, naming and logging
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--name', default = "ID_only_test_GN", type=str,
                    help='name of experiment')
parser.add_argument('--tensorboard',
                    help='Log progress to TensorBoard', action='store_true')
parser.add_argument('--log_name',
                    help='Name of the Log File', type = str, default = "info.log")
parser.add_argument('--base-dir', default='output/ood_scores', type=str, help='result directory')
#Device options
parser.add_argument('--gpu-ids', default='4', type=str,
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
    label_transform = transforms.Compose([ToLabel()])

    if args.in_dataset == "pascal":
        train_set = pascalVOCSet("./datasets/pascal/", split="voc12-train", img_transform = img_transform, 
                                 label_transform = label_transform)
        val_set = pascalVOCSet('./datasets/pascal/', split="voc12-val",
                                   img_transform=val_transform, label_transform=label_transform)      
        # train_loaders = [torch.utils.data.DataLoader(sub_dataset, batch_size=args.batch_size, shuffle=True) 
        #                         for sub_dataset in torch.utils.data.random_split( train_set, [2858, 2859])]
        num_classes = 20
        # pool_size = args.pool_size
        # lr_schedule=[50, 75, 90]
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, num_workers= 4, shuffle=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, num_workers= 4, shuffle=False, pin_memory=True)

    out_dataset = 'imagenet'
    if out_dataset == 'imagenet':
            ood_root = "/nobackup-slow/dataset/ImageNet22k/ImageNet-22K"
            # ood_root = "/nobackup-slow/dataset/nus-ood/"
            out_test_data = datasets.ImageFolder(ood_root, transform=img_transform)
            testloaderOut = torch.utils.data.DataLoader(out_test_data, batch_size=args.val_batch_size, num_workers= 4, pin_memory=True)
    
    if args.model_arch == "resnet101":
        orig_resnet = torchvision.models.resnet101(pretrained=True)
        orig_resnet = rn.l_resnet101()
        rn_checkpoint = torch.load("R-101-GN-WS.pth.tar")
        from collections import OrderedDict
        new_checkpoint = OrderedDict([(k[7:], v) for k, v in rn_checkpoint.items()])
        orig_resnet.load_state_dict(new_checkpoint)
        features = list(orig_resnet.children())
        model = nn.Sequential(*features[0:8])
        clsfier = clssimp(2048, num_classes)
        output_dim = 1
        repr_dim = 1024

        # orig_resnet = torchvision.models.resnet101(pretrained=True)
        # features = list(orig_resnet.children())
        # model = nn.Sequential(*features[0:8])
        # clsfier = clssimp(2048, num_classes)
        # output_dim = 1
        # repr_dim = 1024
    

    model = model.cuda()
    clsfier = clsfier.cuda()

    cudnn.benchmark = True

    freeze_bn_affine = True
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            # print(m)
            m.eval()
            if freeze_bn_affine:
                m.weight.requires_grad = False
                m.bias.requires_grad = False


    criterion = nn.BCEWithLogitsLoss().cuda() # sigmoid 

    optimizer = torch.optim.Adam([{'params': model.parameters(),'lr':args.lr/10},
                                {'params': clsfier.parameters()}], lr=args.lr)
    #CORE
    best_map = 0.0
    mAPs = []
    for epoch in range(args.start_epoch, args.epochs):                                    
        train(args, model, clsfier, train_loader, criterion, optimizer, epoch, directory, log)
        mAP = validate_energy(args, model, clsfier, val_loader, epoch, log)
        if mAP > best_map:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict_model': model.state_dict(),
                'state_dict_clsfier': clsfier.state_dict(),
            }, epoch + 1)
            best_map = mAP
            mAPs.append(mAP)
            log.debug("Epoch [%d/%d][saved] mAP: %.4f" % (epoch + 1, args.epochs, mAP))

        else:
            if epoch % 5 == 0:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict_model': model.state_dict(),
                    'state_dict_clsfier': clsfier.state_dict(),
                }, epoch + 1)
            best_map = mAP
            mAPs.append(mAP)
            log.debug("Epoch [%d/%d][saved] mAP: %.4f" % (epoch + 1, args.epochs, mAP))
            print("Epoch [%d/%d][----] mAP: %.4f" % (epoch + 1, args.epochs, mAP))

    torch.save(torch.tensor(mAPs), os.path.join(directory, "all_mAPs.data") )  

    # checkpoint = torch.load("./checkpoints/{in_dataset}/{name}/checkpoint_{epochs}.pth.tar".format(in_dataset=args.in_dataset, name=args.name, epochs=args.test_epoch))
    # model.load_state_dict(checkpoint['state_dict_model'])
    # clsfier.load_state_dict(checkpoint['state_dict_clsfier'])

    # print("processing ID")
    # id_energy_2, id_energy, id_sum_energy, id_energy_min = get_energy(args, model, clsfier, val_loader, 0, log)
    # print("processing OOD")
    # ood_energy_2, ood_energy, ood_sum_energy, ood_energy_min = get_energy(args, model, clsfier, testloaderOut, 0, log)
    # with open('test.npy', 'wb') as f:
    #     np.save(f,id_energy_2)
    #     np.save(f, id_energy)
    #     np.save(f, id_sum_energy)
    #     np.save(f, ood_energy_2)
    #     np.save(f,  ood_energy)
    #     np.save(f, ood_sum_energy)
    #     np.save(f, id_energy_min)
    #     np.save(f, ood_energy_min)

    # base_dir = args.base_dir
    # in_save_dir = os.path.join(base_dir, args.in_dataset, "energy sum", args.name, 'nat')
    # if not os.path.exists(in_save_dir):
    #     os.makedirs(in_save_dir)
    
    # out_save_dir = os.path.join(in_save_dir, out_dataset)
    # if not os.path.exists(out_save_dir):
    #         os.makedirs(out_save_dir)

    # with open(os.path.join(in_save_dir, "in_scores.txt"), 'w') as f1:
    #     for score in id_sum_energy:
    #         f1.write("{}\n".format(score))
    # with open(os.path.join(out_save_dir, "out_scores.txt"), 'w') as f2:
    #     for score in ood_sum_energy:
    #         f2.write("{}\n".format(score))




def train(args, model, clsfier, train_loader, criterion, optimizer, epoch, save_dir, log):
    """Train for one epoch on the training set"""
    batch_time = AverageMeter()

    in_losses = AverageMeter()
    in_energy_losses = AverageMeter()
    in_energy_all = AverageMeter()
    log.debug("######## Start training NN at epoch {} ########".format(epoch) )

    end = time.time()

    # epoch_buffer_in = torch.empty(0, 3, 224, 224)
    # epoch_buffer_out = torch.empty(0, 3, 224, 224)

    for i, (input, target) in enumerate(train_loader):
        input = input.cuda()
        target = target.cuda().float()
        in_len = len(input)
        model.train()
        clsfier.train()       
        output = model(input)
        output = clsfier(output)
        in_energy = -1 * torch.sum(torch.log(1+torch.exp(output)), dim = 1)
        in_loss = criterion(output, target)  
        in_energy_loss = torch.pow(F.relu(in_energy-args.m_in), 2).mean() 

        in_losses.update(in_loss.data, in_len) 
        in_energy_losses.update(in_energy_loss.data, in_len)
        in_energy_all.update(in_energy.mean().data, in_len)

        loss = in_loss
        # loss = in_loss + self.args.beta * out_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            log.debug('Epoch: [{0}] Batch#[{1}/{2}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'In Ls {in_loss.val:.4f} ({in_loss.avg:.4f})\t'
                'E_in Ls {in_energy_loss.val:.4f} ({in_energy_loss.avg:.4f})\t'
                'E_in Raw {in_energy_all.val:.4f} ({in_energy_all.avg:.4f})\t'.format(
                    epoch, i, len(train_loader),
                    batch_time=batch_time,
                    in_loss=in_losses, 
                    in_energy_loss = in_energy_losses,
                    in_energy_all=in_energy_all,
                    ))


def validate_energy(args, model, clsfier, val_loader, epoch, log):
    in_energy = AverageMeter()
    model.eval()
    clsfier.eval()
    init = True
    log.debug("######## Start validation ########")
    # gts = {i:[] for i in range(0, num_classes)}
    # preds = {i:[] for i in range(0, num_classes)}
    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            images = images.cuda()
            # extra_col = torch.zeros(len(labels), 1)
            # labels = torch.cat( (labels, extra_col), dim = 1).cuda().float()
            labels = labels.cuda().float()
            outputs = model(images)
            outputs = clsfier(outputs)
            E_y = torch.log(1+torch.exp(outputs))
            energy = -1 * torch.sum(E_y, dim = 1).mean()
            in_energy.update(energy.mean().data, len(labels))  #DEBUG
            outputs = torch.sigmoid(outputs)  
            pred = outputs.squeeze().data.cpu().numpy()    
            # in_cls_confs.update(in_cls_conf.data, len(labels))  #DEBUG
            ground_truth = labels.squeeze().data.cpu().numpy()
            if init:
                all_ground_truth = ground_truth 
                all_pred = pred
                init = False
            else:
                all_ground_truth = np.vstack((all_ground_truth, ground_truth) )
                all_pred = np.vstack((all_pred, pred))

            #DEBUG
            if i % args.print_freq == 0:
                log.debug('Epoch: [{0}] Batch#[{1}/{2}]\t'
                    'In Classifer Energy {in_energy.val:.4f} ({in_energy.avg:.4f})'.format(
                        epoch, i, len(val_loader), in_energy=in_energy))

    FinalMAPs = []
    for j in range(20):
        precision, recall, thresholds = metrics.precision_recall_curve(all_ground_truth[:, j], all_pred[:, j])
        FinalMAPs.append(metrics.auc(recall, precision))
        print(f"class {j} auc: {metrics.auc(recall, precision)}")
    print("final map: ", np.mean(FinalMAPs))

    return np.mean(FinalMAPs)


def get_energy(args, model, clsfier, val_loader, epoch, log):
    in_energy = AverageMeter()
    model.eval()
    clsfier.eval()
    init = True
    log.debug("######## Start collecting energy score ########")
    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            images = images.cuda()
            # labels = labels.cuda().float()
            outputs = model(images)
            outputs = clsfier(outputs)
            E_y = torch.log(1+torch.exp(outputs))
            e_s = -1 * torch.sum(E_y, dim = 1)
            energy = e_s.mean()
            e_m_2 = torch.max(torch.pow(E_y, 2), dim = 1)[0].data.cpu().numpy() 
            e_m = torch.max(E_y, dim = 1)[0].data.cpu().numpy() 
            e_min = torch.min(E_y, dim = 1)[0].data.cpu().numpy() 
            e_s = e_s.data.cpu().numpy() 
            in_energy.update(energy.mean().data, len(labels))  #DEBUG
            if init:
                max_energy_2 = e_m_2
                max_energy = e_m
                sum_energy = e_s
                min_energy = e_min
                init = False
            else:
                max_energy_2 = np.concatenate(( max_energy_2 , e_m_2) )
                max_energy = np.concatenate((max_energy, e_m))
                sum_energy = np.concatenate((sum_energy, e_s))
                min_energy = np.concatenate((min_energy, e_min))

            #DEBUG
            if i % args.print_freq == 0:
                log.debug('Epoch: [{0}] Batch#[{1}/{2}]\t'
                    'In Classifer Energy {in_energy.val:.4f} ({in_energy.avg:.4f})'.format(
                        epoch, i, len(val_loader), in_energy=in_energy))

    return max_energy_2, max_energy, sum_energy, min_energy

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
        self.sum += self.val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == '__main__':
    main()

