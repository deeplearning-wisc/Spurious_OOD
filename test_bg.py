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

from models.fine_tuning_layer import clssimp as clssimp
import models.densenet as dn
import models.wideresnet as wn
import models.resnet as rn
import models.simplenet as sn
from models import CNNModel, res18, res50

from tensorboard_logger import configure, log_value
from torch.distributions.multivariate_normal import MultivariateNormal
from rebias_utils import SimpleConvNet


from torch.utils.data import Sampler, DataLoader
from datasets.color_mnist import get_biased_mnist_dataloader
from datasets.cub_dataset import WaterbirdDataset
from datasets.celebA_dataset import get_celebA_dataloader, get_celebA_ood_dataloader
import cv2
from torch.utils.data.dataloader import default_collate
import utils.svhn_loader as svhn

parser = argparse.ArgumentParser(description='OOD Detection Evaluation based on Energy-score')

parser.add_argument('--exp-name', default="erm_r_0_5_2021-05-25", type=str, help='help identify checkpoint')
parser.add_argument('--in-dataset', default="color_mnist", type=str, help='in-distribution dataset')
parser.add_argument('--model-arch', default='resnet18', type=str, help='model architecture e.g. resnet101')
parser.add_argument('--method', default='dann', type=str, help='method used for model training')
parser.add_argument('--print-freq', '-p', default=10, type=int, help='print frequency (default: 10)') # print every print-freq batches during training
# ID train & val batch size and OOD train batch size 
parser.add_argument('-b', '--batch-size', default= 64, type=int,
                    help='mini-batch size (default: 64) used for training id and ood')
parser.add_argument('--num-classes', default=2, type=int,
                    help='number of classes for model training')
# ood
parser.add_argument('--ood-batch-size', default= 64, type=int,
                    help='mini-batch size (default: 400) used for testing')
parser.add_argument('--data_label_correlation', default= 0.4, type=float,
                    help='data_label_correlation')
parser.add_argument('--name', '-n', default = "irm_test_0.4_3rd", type=str,
                    help='name of experiment')
parser.add_argument('--test_epochs', "-e", default = "10", type=str,
                     help='# epoch to test performance')
parser.add_argument('--log_name',
                    help='Name of the Log File', type = str, default = "info_val.log")
parser.add_argument('--base-dir', default='output/ood_scores', type=str, help='result directory')
parser.add_argument('--CAM', type=bool, default = False, help='if generate CAM')
#Device options
parser.add_argument('--gpu-ids', default='5', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--local_rank', default=-1, type=int,
                        help='rank for the current node')
parser.add_argument('--multi-gpu', default=False, type=bool)
parser.set_defaults(bottleneck=True)
parser.set_defaults(augment=True)

args = parser.parse_args()

state = {k: v for k, v in args._get_kwargs()}
directory = "checkpoints/{in_dataset}/{name}/".format(in_dataset=args.in_dataset, name=args.name)
if not os.path.exists(directory):
    os.makedirs(directory)
save_state_file = os.path.join(directory, 'args.txt')
fw = open(save_state_file, 'w')
print(state, file=fw)
fw.close()

# Use CUDA
# os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
# torch.manual_seed(1)
# np.random.seed(1)
# np.random.seed(1)
# use_cuda = torch.cuda.is_available()
# if use_cuda:
#     torch.cuda.manual_seed_all(1)
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
if torch.cuda.is_available():
    torch.cuda.set_device(args.local_rank)
device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
args.n_gpus =torch.cuda.device_count()
if args.n_gpus > 1:
    import torch.distributed as dist
    import torch.multiprocessing as mp
    from torch.utils.data.distributed import DistributedSampler
    from torch.nn.parallel import DistributedDataParallel as DDP
    # import apex
    # from apex.parallel import DistributedDataParallel as DDP
    # from apex import amp
    args.multi_gpu = True
    torch.distributed.init_process_group(
        'nccl',
        init_method='env://',
        world_size=args.n_gpus,
        rank=args.local_rank,
    )
    # devices = list(range(args.n_gpus))
else:
    args.multi_gpu = False

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

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        ])

    kwargs = {'num_workers': 4, 'pin_memory': True}
    val_transform = transforms.Compose([transforms.Scale(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        normalizer])

    if args.in_dataset == "color_mnist":
        val_loader = get_biased_mnist_dataloader(args, root = './datasets/MNIST', batch_size=args.batch_size,
                                            data_label_correlation= args.data_label_correlation,
                                            n_confusing_labels= args.num_classes - 1,
                                            train=False, partial=True, cmap = "1")
        num_classes = args.num_classes
    elif args.in_dataset == "waterbird":
        val_dataset = WaterbirdDataset(data_correlation=args.data_label_correlation, train=False)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
        num_classes = args.num_classes
    elif args.in_dataset == "celebA":
        val_loader = get_celebA_dataloader(args, train=False)
        num_classes = args.num_classes

    # create model
    if args.model_arch == "general_model":
        base_model = CNNModel(num_classes=args.num_classes, bn_init=True, method=args.method)
    elif args.model_arch == "resnet50":
        if args.in_dataset == "waterbird":
            base_model = res50(n_classes=args.num_classes, method=args.method, domain_num=4)
        else:
            base_model = res50(n_classes=args.num_classes, method=args.method)
    elif args.model_arch == "resnet18":
        base_model = res18(n_classes=args.num_classes, method=args.method)
    else:
        assert False, 'Not supported model arch: {}'.format(args.model_arch)

    model = base_model.cuda()
    # if args.method == "dann" or args.method == "erm" or args.method == "irm":
    #     model = base_model.cuda()
    # elif args.method == "rebias":
    #     n_g_nets = 1
    #     f_model = base_model.cuda()
    #     g_model = [base_model.cuda() for _ in range(n_g_nets)]
    # else:
    #     assert False, 'Not supported method: {}'.format(args.method)

    test_epochs = args.test_epochs.split()
    if args.in_dataset == 'color_mnist':
        out_datasets = ['partial_color_mnist_0&1']
        #out_datasets = ['dtd', 'iSUN', 'LSUN_resize']
         # out_datasets = ['partial_color_mnist']
    elif args.in_dataset == 'waterbird':
        out_datasets = ['placesbg']
    elif args.in_dataset == 'color_mnist_multi':
        out_datasets = ['partial_color_mnist_0&1']
    # load model and store test results

    for test_epoch in test_epochs:
        # print("./checkpoints/{in_dataset}/{name}/checkpoint_{epochs}.pth.tar".format(in_dataset=args.in_dataset, name=args.name, epochs= test_epoch))
        # checkpoint = torch.load("./checkpoints/{in_dataset}/{name}/checkpoint_{epochs}.pth.tar".format(in_dataset=args.in_dataset, name=args.name, epochs= test_epoch))
        # model.load_state_dict(checkpoint['state_dict'])
        cpts_directory = "/nobackup/spurious_ood/checkpoints/{in_dataset}/{name}/{exp}".format(in_dataset=args.in_dataset, name=args.name, exp=args.exp_name)
        cpts_dir = os.path.join(cpts_directory, "checkpoint_{epochs}.pth.tar".format(epochs=test_epoch))
        checkpoint = torch.load(cpts_dir)
        model.load_state_dict(checkpoint['state_dict_model'])
        model.eval()
        model.cuda()
        # if args.model_arch != "rebias_conv" and args.model_arch != "dann":
        #     clsfier.load_state_dict(checkpoint['state_dict_clsfier'])
        #     clsfier.eval()
        #     clsfier.cuda()
        save_dir =  f"./energy_results/{args.in_dataset}/{args.name}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        print("processing ID")
        # id_sum_energy, id_cmt = get_energy(args, model, clsfier, val_loader, test_epoch, log, id = False)
        # id_sum_energy = get_energy_biased(args, model, clsfier, val_loader, test_epoch, log, id = False)
        id_sum_energy,id_cmt = get_energy_biased(args, model, val_loader, test_epoch, log, method=args.method, id = True)
        with open(os.path.join(save_dir, f'energy_score_at_epoch_{test_epoch}.npy'), 'wb') as f:
            np.save(f, id_sum_energy)
            # np.save(f, id_cmt)
        for out_dataset in out_datasets:
            print("processing OOD dataset ", out_dataset)
            testloaderOut = get_ood_loader(out_dataset)
            ood_sum_energy = get_energy(args, model, testloaderOut, test_epoch, log, method=args.method, id = False)
            # ood_sum_energy, ood_cmt = get_energy(args, model, clsfier, testloaderOut, test_epoch, log, id = True)
            with open(os.path.join(save_dir, f'energy_score_{out_dataset}_at_epoch_{test_epoch}.npy'), 'wb') as f:
                np.save(f, ood_sum_energy)
                # np.save(f, ood_cmt)

def get_energy(args, model, val_loader, epoch, log, method, id = False, CAM = False):
    in_energy = AverageMeter()
    top1 = AverageMeter()
    model.eval()
    init = True
    log.debug("######## Start collecting energy score ########")
    with torch.no_grad():
        if id:
            all_preds = torch.tensor([])
            all_targets = torch.tensor([])
        for i, (images, labels) in enumerate(val_loader):
            images = images.cuda()
            if method == "dann":
                _, outputs, _ = model(images, alpha=0)
            else:
                _, outputs = model(images)
            if id:
                all_targets = torch.cat((all_targets, labels),dim=0)
                all_preds = torch.cat((all_preds, outputs.argmax(dim=1).cpu()),dim=0)
                prec1 = accuracy(outputs.data, labels, topk=(1,))[0]
                top1.update(prec1, images.size(0))
            e_s = -torch.logsumexp(outputs, dim=1)
            e_s = e_s.data.cpu().numpy() 
            in_energy.update(e_s.mean(), len(labels))  #DEBUG
            if init:
                sum_energy = e_s
                init = False
            else:
                sum_energy = np.concatenate((sum_energy, e_s))

            #DEBUG
            if i % args.print_freq == 0: 
                log.debug('Epoch: [{0}] Batch#[{1}/{2}]\t'
                    'Energy Sum {in_energy.val:.4f} ({in_energy.avg:.4f})'.format(
                        epoch, i, len(val_loader), in_energy=in_energy))
        if id:
            log.debug(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
            stacked = torch.stack((all_targets,all_preds),dim=1)
            cmt = torch.zeros(9,9, dtype=torch.int64)
            for p in stacked:
                tl, pl = p.type(torch.int64).tolist()
                cmt[tl, pl] = cmt[tl, pl] + 1
            return sum_energy, cmt
        else:
            return sum_energy

def get_energy_biased(args, model, val_loader, epoch, log, method, id = False, CAM = False):
    in_energy = AverageMeter()
    top1 = AverageMeter()
    model.eval()
    init = True
    log.debug("######## Start collecting energy score ########")
    with torch.no_grad():
        if id:
            all_preds = torch.tensor([])
            all_targets = torch.tensor([])
        for i, (images, labels, _) in enumerate(val_loader):
            images = images.cuda()
            if method == "dann":
                _, outputs, _ = model(images, alpha=0)
            else:
                _, outputs = model(images)
            if id:
                all_targets = torch.cat((all_targets, labels),dim=0)
                all_preds = torch.cat((all_preds, outputs.argmax(dim=1).cpu()),dim=0)
                prec1 = accuracy(outputs.cpu().data, labels, topk=(1,))[0]
                top1.update(prec1, images.size(0))
            e_s = -torch.logsumexp(outputs, dim=1)
            e_s = e_s.data.cpu().numpy() 
            in_energy.update(e_s.mean(), len(labels))  #DEBUG
            if init:
                sum_energy = e_s
                init = False
            else:
                sum_energy = np.concatenate((sum_energy, e_s))

            #DEBUG
            if i % args.print_freq == 0: 
                log.debug('Epoch: [{0}] Batch#[{1}/{2}]\t'
                    'Energy Sum {in_energy.val:.4f} ({in_energy.avg:.4f})'.format(
                        epoch, i, len(val_loader), in_energy=in_energy))
        if id:
            log.debug(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
            stacked = torch.stack((all_targets,all_preds),dim=1)
            cmt = torch.zeros(5,5, dtype=torch.int64)
            for p in stacked:
                tl, pl = p.type(torch.int64).tolist()
                cmt[tl, pl] = cmt[tl, pl] + 1
            return sum_energy, cmt
        else:
            return sum_energy

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

def get_ood_loader(out_dataset, CAM = False):
        # normalizer = transforms.Normalize((125.3/255, 123.0/255, 113.9/255), (63.0/255, 62.1/255.0, 66.7/255.0))
        # val_transform  =  transforms.Compose([transforms.Scale(256),
        #                                 transforms.CenterCrop(224),
        #                                 transforms.ToTensor(),
        #                                 normalizer])
        val_transform = transforms.Compose([
                transforms.Resize(28),
                 transforms.CenterCrop(28),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5),
                             std=(0.5, 0.5, 0.5))])
        if out_dataset == 'SVHN':
            testsetout = svhn.SVHN('datasets/ood_datasets/svhn/', split='test',
                                  transform= val_transform, download=False)
            testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=args.ood_batch_size,
                                             shuffle=True, num_workers=2)
        elif out_dataset == 'dtd':
            testsetout = torchvision.datasets.ImageFolder(root="datasets/ood_datasets/dtd/images",
                                        transform=val_transform)
            testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=args.ood_batch_size, shuffle=True,
                                                     num_workers=2)
        elif out_dataset == 'places365':
            testsetout = torchvision.datasets.ImageFolder(root="/nobackup-slow/dataset/places365",
                transform=val_transform)
            subset = torch.utils.data.Subset(testsetout, np.random.choice(len(testsetout), 5000, replace=False))
            testloaderOut = torch.utils.data.DataLoader(subset, batch_size=args.ood_batch_size,
                                                     num_workers=2, shuffle=True)
        elif 'partial_color_mnist' in out_dataset or out_dataset == "0_1_cross" or out_dataset == "0_background":
            val_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5),
                             std=(0.5, 0.5, 0.5))])
            testsetout = torchvision.datasets.ImageFolder("./datasets/ood_datasets/{}".format(out_dataset),
                                        transform=val_transform)
            testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=args.ood_batch_size,
                                             shuffle=True, num_workers=2)
        elif 'placesbg' in out_dataset:
            scale = 256.0/224.0
            target_resolution = (224, 224)
            val_transform = transforms.Compose([
                transforms.Resize((int(target_resolution[0]*scale), int(target_resolution[1]*scale))),
                transforms.CenterCrop(target_resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            testsetout = torchvision.datasets.ImageFolder("./datasets/ood_datasets/{}".format(out_dataset),
                                        transform=val_transform)
            subset = torch.utils.data.Subset(testsetout, np.random.choice(len(testsetout), 5000, replace=False))
            testloaderOut = torch.utils.data.DataLoader(subset, batch_size=args.ood_batch_size,
                                             shuffle=True, num_workers=2)           
        else:
            testsetout = torchvision.datasets.ImageFolder("./datasets/ood_datasets/{}".format(out_dataset),
                                        transform= val_transform)
            testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=args.ood_batch_size,
                                             shuffle=True, num_workers=2)
        return testloaderOut

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

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

if __name__ == '__main__':
    main()

