import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
# os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"
# os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"
import sys

import shutil
import time

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
import numpy as np
import utils.svhn_loader as svhn

import models.densenet as dn
import models.wideresnet as wn
import models.resnet as rn
import models.simplenet as sn
from neural_linear import NeuralLinear
from utils import LinfPGDAttack, TinyImages
# used for logging to TensorBoard
from tensorboard_logger import configure, log_value
from torch.distributions.multivariate_normal import MultivariateNormal

import json
import logging

from torch.utils.data import Sampler
from neural_linear import SimpleDataset


parser = argparse.ArgumentParser(description='PyTorch DenseNet Training')
# parser.add_argument('--gpu', default=3, type=int, help='the preferred gpu to use')

parser.add_argument('--in-dataset', default="CIFAR-10", type=str, help='in-distribution dataset e.g. CIFAR-10')
parser.add_argument('--model-arch', default='densenet', type=str, help='model architecture e.g. simplenet densenet')

parser.add_argument('--save-epoch', default= 10, type=int,
                    help='save the model every save_epoch') # freq; save model state_dict()
parser.add_argument('--save-data-epoch', default= 100, type=int,
                    help='save the sampled ood every save_data_epoch') # freq; save data
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    help='print frequency (default: 10)') # print every print-freq batches during training

# ID train & val batch size and OOD train batch size 
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    help='mini-batch size (default: 64) used for training id and ood')

# training schedule
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--epochs', default= 100, type=int,
                    help='number of total epochs to run')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', '--wd', default=0.0001, type=float,
                    help='weight decay (default: 0.0001)')

# densenet
parser.add_argument('--layers', default= 100, type=int,
                    help='total number of layers (default: 100) for DenseNet')
parser.add_argument('--growth', default=12, type=int,
                    help='number of new channels per layer (default: 12)')
# wideresnet
parser.add_argument('--depth', default=40, type=int,
                    help='depth of wide resnet')
parser.add_argument('--width', default=4, type=int,
                    help='width of resnet')

## network spec
parser.add_argument('--droprate', default=0.0, type=float,
                    help='dropout probability (default: 0.0)')
parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='whether to use standard augmentation (default: True)')
parser.add_argument('--reduce', default=0.5, type=float,
                    help='compression rate in transition stage (default: 0.5)')
parser.add_argument('--no-bottleneck', dest='bottleneck', action='store_false',
                    help='To not use bottleneck block')
parser.add_argument('--beta', default=1.0, type=float, help='beta for out_loss')

# ood sampling and mining
parser.add_argument('--ood-batch-size', default= 400, type=int,
                    help='mini-batch size (default: 400) used for ood mining')
parser.add_argument('--pool-size', default= 1000, type=int,
                    help='pool size')
# parser.add_argument('--ood_dataset_size', type= int, default= 75000,
#                  help='ood_dataset_size per epoch. 750 for 3-class-gaussian')

#posterior sampling
parser.add_argument('--a0', type=float, default=6.0, help='a0')
parser.add_argument('--b0', type=float, default=6.0, help='b0')
parser.add_argument('--lambda_prior', type=float, default=0.25, help='lambda_prior')
parser.add_argument('--sigma', type=float, default= 100, help='control var for weights')
parser.add_argument('--sigma_n', type=float, default=1, help='control var for noise')
parser.add_argument('--conf', type=float, default=4.6, help='control ground truth for bayesian linear regression.3.9--0.98; 4.6 --0.99')

# saving, naming and logging
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--name', default = "4_ntom_seperate", type=str,
                    help='name of experiment')
parser.add_argument('--tensorboard',
                    help='Log progress to TensorBoard', action='store_true')
parser.add_argument('--log_name',
                    help='Name of the Log File', type = str, default = "info.log")

#deprecated
# parser.add_argument('--total_sample_times', default = 1, type=int,
#                     help='total sample times per epoch for OOD') 
parser.add_argument('--ood_factor', type=float, default= 2,
                 help='ood_dataset_size = len(train_loader.dataset) * ood_factor default = 2.0')

parser.set_defaults(bottleneck=True)
parser.set_defaults(augment=True)

args = parser.parse_args()
devices = list(range(torch.cuda.device_count()))

state = {k: v for k, v in args._get_kwargs()}
print(state)
directory = "checkpoints/{in_dataset}/{name}/".format(in_dataset=args.in_dataset, name=args.name)
if not os.path.exists(directory):
    os.makedirs(directory)
save_state_file = os.path.join(directory, 'args.txt')
fw = open(save_state_file, 'w')
print(state, file=fw)
fw.close()


torch.manual_seed(1)
np.random.seed(1)

def main():
    if args.tensorboard: configure("runs/%s"%(args.name))

    if args.augment:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        ])

    kwargs = {'num_workers': 1, 'pin_memory': True}

    # setup_logger(args.log_name, os.path.join(directory, args.log_name))
    log = logging.getLogger(__name__)
    formatter = logging.Formatter('%(asctime)s : %(message)s')
    fileHandler = logging.FileHandler(os.path.join(directory, args.log_name), mode='w')
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)
    log.setLevel(logging.DEBUG)
    log.addHandler(fileHandler)
    log.addHandler(streamHandler) 
    if args.in_dataset == "CIFAR-10":
        # Data loading code
        normalizer = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x/255.0 for x in [63.0, 62.1, 66.7]])
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./datasets/cifar10', train=True, download=True,
                             transform=transform_train),
            batch_size=args.batch_size, shuffle=True, **kwargs)

        # train_loaders = [torch.utils.data.DataLoader(sub_dataset, batch_size=args.batch_size, shuffle=True, **kwargs) 
        #                         for sub_dataset in torch.utils.data.random_split( datasets.CIFAR10('./datasets/cifar10', train=True, download=True,
        #                      transform=transform_train), [25000]*2)]
        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./datasets/cifar10', train=False, transform=transform_test),
            batch_size=args.batch_size, shuffle=True, **kwargs)

        lr_schedule=[50, 75, 90]

        num_classes = 10
        pool_size = args.pool_size

    elif args.in_dataset == "CIFAR-100":
        # Data loading code
        normalizer = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x/255.0 for x in [63.0, 62.1, 66.7]])
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('./datasets/cifar100', train=True, download=True,
                             transform=transform_train),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('./datasets/cifar100', train=False, transform=transform_test),
            batch_size=args.batch_size, shuffle=True, **kwargs)

        lr_schedule=[50, 75, 90]

        num_classes = 100
        pool_size = args.pool_size

    elif args.in_dataset == "SVHN":
        # Data loading code
        normalizer = None
        train_loader = torch.utils.data.DataLoader(
            svhn.SVHN('datasets/svhn/', split='train',
                                      transform=transforms.ToTensor(), download=False),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        # train_loaders = [torch.utils.data.DataLoader(sub_dataset, batch_size=args.batch_size, shuffle=True, **kwargs) 
        #                         for sub_dataset in torch.utils.data.random_split( datasets.CIFAR10('./datasets/svhn', train=True, download=True,
        #                      transform=transform_train), [36628, 36629])]
        train_loaders = [torch.utils.data.DataLoader(sub_dataset, batch_size=args.batch_size, shuffle=True, **kwargs) 
                                for sub_dataset in torch.utils.data.random_split( svhn.SVHN('datasets/svhn/', split='train',
                                      transform=transforms.ToTensor(), download=False), [18314,18314,18314,18315])]
                             
        val_loader = torch.utils.data.DataLoader(
            svhn.SVHN('datasets/svhn/', split='test',
                                  transform=transforms.ToTensor(), download=False),
            batch_size=args.batch_size, shuffle=False, **kwargs)

        args.epochs = 30
        args.save_epoch = 5
        lr_schedule=[10, 15, 18]
        # pool_size = int(len(train_loader.dataset) * 8 / args.ood_batch_size) + 1
        pool_size = args.pool_size
        num_classes = 10

    ood_dataset_size = int(len(train_loader.dataset) * args.ood_factor)

    # ood_dataset_size = args.ood_dataset_size
    # ood_dataset_size = len(train_loader.dataset) * 2
    print('OOD Dataset Size: ', ood_dataset_size)

    ood_loader = torch.utils.data.DataLoader(
        TinyImages(transform=transforms.Compose(
            [transforms.ToTensor(), transforms.ToPILImage(), transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(), transforms.ToTensor()])),
            batch_size=args.ood_batch_size, shuffle=False, **kwargs)

    # create model
    if args.model_arch == 'densenet':
        model = dn.DenseNet3(args.layers, num_classes + 1, args.growth, reduction=args.reduce,
                             bottleneck=args.bottleneck, dropRate=args.droprate, normalizer=normalizer)
    elif args.model_arch == 'wideresnet':
        model = wn.WideResNet(args.depth, num_classes + 1, widen_factor=args.width, dropRate=args.droprate, normalizer=normalizer)
    elif args.model_arch == "resnet":
        model = rn.ResNet20(num_classes = num_classes + 1)
    elif args.model_arch == "simplenet":
        model = sn.SimpleNet(num_classes = num_classes + 1)
    else:
        assert False, 'Not supported model arch: {}'.format(args.model_arch)

    # get the number of model parameters
    # print('Number of model parameters: {}'.format(
    #     sum([p.data.nelement() for p in model.parameters()])))

    output_dim = 1
    repr_dim = model.repr_dim
    print("repr dim: ", repr_dim)
    model = model.cuda()
    bayes_nn = NeuralLinear(args, model, repr_dim, output_dim)

    cudnn.benchmark = True

    # define loss function (criterion) and pptimizer
    criterion = nn.CrossEntropyLoss().cuda()
    # sigmoid 
    # criterion = nn.BCEWithLogitsLoss().cuda()


    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                nesterov=True,
                                weight_decay=args.weight_decay)

    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001, 
    #                 betas=(0.9, 0.999), 
    #                 eps=1e-08, 
    #                 weight_decay=args.weight_decay, 
    #                 amsgrad=False)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            # checkpoint = torch.load(args.resume, map_location = 'cuda:0') # (if not Dataparallel) loads the model to a given GPU device
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            # model = model.to('cuda:0') # (if not Dataparallel) convert the model’s parameter tensors to CUDA tensors
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    #CORE
    bayes_nn.sample_BDQN()
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, lr_schedule)
        print("in every epoch, the total ood pool size is : ", args.ood_batch_size * args.pool_size )
        print("will sample {} ood from it for training".format(ood_dataset_size))

        #atom + ours DEBUG
        # selected_ood_loader = select_ood_atom(ood_loader, model, bayes_nn, args.batch_size * 1, 
        #                                      num_classes, pool_size, epoch, ood_dataset_size, log)
        #ours
        # selected_ood_loader = select_ood(ood_loader, model, bayes_nn, args.batch_size * 1, 
        #                                      num_classes, pool_size, epoch, ood_dataset_size, log)
        # bayes_nn.train_blr(train_loader, selected_ood_loader, model, criterion, num_classes, optimizer, epoch, directory, log)
        # bayes_nn.update_representation()
        # bayes_nn.update_bays_reg_BDQN(log)
        # bayes_nn.sample_BDQN()

        # multi-pass
        # for sub_train_loader in train_loaders: 
        #     selected_ood_loader = select_ood(ood_loader, model, bayes_nn, args.batch_size * 2, 
        #                                      num_classes, pool_size, epoch, ood_dataset_size, use_ATOM= False)
        #     bayes_nn.train_blr(sub_train_loader, selected_ood_loader, model, criterion, num_classes, optimizer, epoch, directory)
        #     bayes_nn.update_representation()
        #     bayes_nn.update_bays_reg_BDQN()
        #     bayes_nn.sample_BDQN()

        selected_ood_loader = select_ood_atom_clean(ood_loader, model, bayes_nn, args.batch_size * 2, 
                                             num_classes, pool_size, epoch, ood_dataset_size, log)
        bayes_nn.train_blr(train_loader, selected_ood_loader, model, criterion, num_classes, optimizer, epoch, directory, log)


        # evaluate on validation set
        prec1 =  bayes_nn.validate(val_loader, model, criterion, epoch, log)
        #DEBUG
        log.debug("cov and mean checking...")
        log.debug(f"cov: {bayes_nn.cov_w[0,:6,:6]}")
        log.debug(f"mu: {bayes_nn.mu_w[0][:20]}")

        # remember best prec@1 and save checkpoint
        if (epoch + 1) % args.save_epoch == 0:
            # data parallel save
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
            }, epoch + 1)
            

def select_ood(ood_loader, model, bayes_model, batch_size, num_classes, pool_size, epoch, ood_dataset_size, log):

    # start at a random point of the outlier dataset; this induces more randomness without obliterating locality
    offset = np.random.randint(len(ood_loader.dataset))
    while offset>=0 and offset<10000:
        offset = np.random.randint(len(ood_loader.dataset))

    ood_loader.dataset.offset = offset

    out_iter = iter(ood_loader)

    log.debug('######## Start selecting OOD samples ########')
    start = time.time()

    # select ood samples
    model.eval()
    with torch.no_grad():
        all_ood_input = []
        all_abs_val = []
        all_ood_conf = []
        softmax_all_ood_conf = []
        # sigmoid_all_ood_conf = []
        for k in range(pool_size): 
            if k % 50 == 0:
                print("ood selection batch ", k)
            try:
                out_set = next(out_iter)
            except StopIteration:
                offset = np.random.randint(len(ood_loader.dataset))
                while offset>=0 and offset<10000:
                    offset = np.random.randint(len(ood_loader.dataset))
                ood_loader.dataset.offset = offset
                out_iter = iter(ood_loader)
                out_set = next(out_iter)

            input = out_set[0] # 128, 3, 32, 32
            output = bayes_model.predict(input.cuda())
            abs_val = torch.abs(output).squeeze()  
            prob = torch.sigmoid(output)

            #softmax output
            softmax_classifier_output = model(input.cuda())
            softmax_prob = F.softmax(softmax_classifier_output, dim=1)
            softmax_all_ood_conf.extend(softmax_prob[:,-1].unsqueeze(-1).detach().cpu().numpy())

            all_ood_input.append(input)
            all_ood_conf.extend(prob.detach().cpu().numpy())
            all_abs_val.extend(abs_val.detach().cpu().numpy())

    all_ood_input = torch.cat(all_ood_input, 0)
    all_abs_val = np.array(all_abs_val) # (400000,)
    all_ood_conf = np.array(all_ood_conf)
    argmin_abs_val = np.argsort(all_abs_val) 
    softmax_all_ood_conf = np.array(softmax_all_ood_conf).squeeze()
    # if epoch < 10 and use_ATOM: 
    #     print('select according to ATOM')
    #     N = all_ood_input.shape[0]
    #     quantile = 0.125
    #     indices = np.argsort(softmax_all_ood_conf )
    #     selected_indices = indices[int(quantile*N):int(quantile*N)+ood_dataset_size]
    # else:
    selected_indices = argmin_abs_val[: ood_dataset_size]
    selected_ood_conf = all_ood_conf[selected_indices]
    log.debug(f'Max OOD Conf: {np.max(all_ood_conf)} Min OOD Conf: {np.min(all_ood_conf)} Average OOD Conf: {np.mean(all_ood_conf)}')
    log.debug(f'Selected Max OOD Conf: {np.max(selected_ood_conf)} Selected Min OOD Conf: {np.min(selected_ood_conf)} Selected Average OOD Conf: {np.mean(selected_ood_conf)}')
    softmax_selected_ood_conf = softmax_all_ood_conf[selected_indices]
    print('Total OOD samples: ', len(selected_indices))
    log.debug(f'(Softmax classifier) Max OOD Conf: {np.max(softmax_all_ood_conf)} Min OOD Conf: {np.min(softmax_all_ood_conf)} Average OOD Conf: {np.mean(softmax_all_ood_conf)}')
    log.debug(f'(Softmax classifier) Selected Max OOD Conf: {np.max(softmax_selected_ood_conf)} Selected Min OOD Conf: {np.min(softmax_selected_ood_conf)} Selected Average OOD Conf: {np.mean(softmax_selected_ood_conf)}')
    ood_images = all_ood_input[selected_indices]
    ood_labels = (torch.ones(ood_dataset_size) * num_classes).long()
    ood_train_loader = torch.utils.data.DataLoader(
        OODDataset(ood_images, ood_labels),
        batch_size=batch_size, shuffle=True)

    print('Time: ', time.time()-start)

    return ood_train_loader

def select_ood_atom_clean(ood_loader, model, bayes_model, batch_size, num_classes, pool_size, epoch, ood_dataset_size, log):

    # start at a random point of the outlier dataset; this induces more randomness without obliterating locality
    offset = np.random.randint(len(ood_loader.dataset))
    while offset>=0 and offset<10000:
        offset = np.random.randint(len(ood_loader.dataset))

    out_iter = iter(ood_loader)

    log.debug('######## Start selecting OOD samples ########')
    start = time.time()

    # select ood samples
    model.eval()
    with torch.no_grad():
        all_ood_input = []
      
        softmax_all_ood_conf = []
        for k in range(pool_size): 
            if k % 50 == 0:
                print("ood selection batch ", k)
            try:
                out_set = next(out_iter)
            except StopIteration:
                offset = np.random.randint(len(ood_loader.dataset))
                while offset>=0 and offset<10000:
                    offset = np.random.randint(len(ood_loader.dataset))
                ood_loader.dataset.offset = offset
                out_iter = iter(ood_loader)
                out_set = next(out_iter)

            input = out_set[0] # 128, 3, 32, 32
            #softmax outputxw
            softmax_classifier_output = model(input.cuda())
            softmax_conf = F.softmax(softmax_classifier_output, dim=1)[:,-1]
            softmax_all_ood_conf.extend(softmax_conf.detach().cpu().numpy())
            all_ood_input.append(input)

    all_ood_input = torch.cat(all_ood_input, 0)
    softmax_all_ood_conf = np.array(softmax_all_ood_conf).squeeze()
    # ood selection 
    N = all_ood_input.shape[0]
    quantile = 0.125
    indices = np.argsort(softmax_all_ood_conf )
    
    selected_indices = indices[int(quantile*N):int(quantile*N)+ood_dataset_size]
    # selected_indices = np.random.choice(N, ood_dataset_size, replace = False)
    softmax_selected_ood_conf = softmax_all_ood_conf[selected_indices]
    print('Total OOD samples: ', len(selected_indices))
    log.debug(f'(Softmax classifier) Max OOD Conf: {np.max(softmax_all_ood_conf)} Min OOD Conf: {np.min(softmax_all_ood_conf)} Average OOD Conf: {np.mean(softmax_all_ood_conf)}')
    log.debug(f'(Softmax classifier) Selected Max OOD Conf: {np.max(softmax_selected_ood_conf)} Selected Min OOD Conf: {np.min(softmax_selected_ood_conf)} Selected Average OOD Conf: {np.mean(softmax_selected_ood_conf)}')
    ood_images = all_ood_input[selected_indices]
    ood_labels = (torch.ones(ood_dataset_size) * num_classes).long()
    ood_train_loader = torch.utils.data.DataLoader(
        OODDataset(ood_images, ood_labels),
        batch_size=batch_size, shuffle=True)

    print('Time: ', time.time()-start)

    return ood_train_loader

def select_ood_atom(ood_loader, model, bayes_model, batch_size, num_classes, pool_size, epoch, ood_dataset_size, log):

    # start at a random point of the outlier dataset; this induces more randomness without obliterating locality
    offset = np.random.randint(len(ood_loader.dataset))
    while offset>=0 and offset<10000:
        offset = np.random.randint(len(ood_loader.dataset))

    out_iter = iter(ood_loader)

    log.debug('######## Start selecting OOD samples ########')
    start = time.time()

    # select ood samples
    model.eval()
    with torch.no_grad():
        all_ood_input = []
        all_ood_conf = []  #DEBUG
        softmax_all_ood_conf = []
        for k in range(pool_size): 
            if k % 50 == 0:
                print("ood selection batch ", k)
            try:
                out_set = next(out_iter)
            except StopIteration:
                offset = np.random.randint(len(ood_loader.dataset))
                while offset>=0 and offset<10000:
                    offset = np.random.randint(len(ood_loader.dataset))
                ood_loader.dataset.offset = offset
                out_iter = iter(ood_loader)
                out_set = next(out_iter)

            input = out_set[0] # 128, 3, 32, 32
            #softmax output
            output = bayes_model.predict(input.cuda()) #DEBUG
            prob = torch.sigmoid(output)              #DEBUG

            all_ood_conf.extend(prob.detach().cpu().numpy()) #DEBUG
            softmax_classifier_output = model(input.cuda())
            softmax_conf = F.softmax(softmax_classifier_output, dim=1)[:,-1]
            softmax_all_ood_conf.extend(softmax_conf.detach().cpu().numpy())
            all_ood_input.append(input)

    all_ood_input = torch.cat(all_ood_input, 0)
    softmax_all_ood_conf = np.array(softmax_all_ood_conf).squeeze()
    all_ood_conf = np.array(all_ood_conf) #DEBUG

    # ood selection 
    N = all_ood_input.shape[0]
    quantile = 0.125
    indices = np.argsort(softmax_all_ood_conf )
    selected_indices = indices[int(quantile*N):int(quantile*N)+ood_dataset_size]
    softmax_selected_ood_conf = softmax_all_ood_conf[selected_indices]
    selected_ood_conf = all_ood_conf[selected_indices]  #DEBUG
    log.debug(f'Max OOD Conf: {np.max(all_ood_conf)} Min OOD Conf: {np.min(all_ood_conf)} Average OOD Conf: {np.mean(all_ood_conf)}')
    log.debug(f'Selected Max OOD Conf: {np.max(selected_ood_conf)} Selected Min OOD Conf: {np.min(selected_ood_conf)} Selected Average OOD Conf: {np.mean(selected_ood_conf)}')
    print('Total OOD samples: ', len(selected_indices))
    log.debug(f'(Softmax classifier) Max OOD Conf: {np.max(softmax_all_ood_conf)} Min OOD Conf: {np.min(softmax_all_ood_conf)} Average OOD Conf: {np.mean(softmax_all_ood_conf)}')
    log.debug(f'(Softmax classifier) Selected Max OOD Conf: {np.max(softmax_selected_ood_conf)} Selected Min OOD Conf: {np.min(softmax_selected_ood_conf)} Selected Average OOD Conf: {np.mean(softmax_selected_ood_conf)}')
    ood_images = all_ood_input[selected_indices]
    ood_labels = (torch.ones(ood_dataset_size) * num_classes).long()
    ood_train_loader = torch.utils.data.DataLoader(
        OODDataset(ood_images, ood_labels),
        batch_size=batch_size, shuffle=True)

    print('Time: ', time.time()-start)

    return ood_train_loader

def select_ood_random(ood_loader, model, batch_size, num_classes, pool_size, ood_dataset_size = None):

    # start at a random point of the outlier dataset; this induces more randomness without obliterating locality
    ood_loader.dataset.offset = np.random.randint(len(ood_loader.dataset))

    out_iter = iter(ood_loader)

    print('Start selecting OOD samples randomly...')
    start = time.time()

    # select ood samples
    model.eval()
    with torch.no_grad():
        all_ood_input = []
        all_ood_conf = []
        softmax_all_ood_conf = []
        for k in range(pool_size): 
            # if k % 10 == 0:
            #     print("ood selection batch ", k)
            try:
                out_set = next(out_iter)
            except StopIteration:
                out_iter = iter(ood_loader)
                out_set = next(out_iter)

            input = out_set[0] # 128, 3, 32, 32
            output = model(input.cuda())
            softmax_prob = F.softmax(output, dim=1)
            prob = torch.sigmoid(output)

            all_ood_input.append(input)
            all_ood_conf.extend(prob[:,-1].unsqueeze(-1).detach().cpu().numpy())
            softmax_all_ood_conf.extend(softmax_prob[:,-1].unsqueeze(-1).detach().cpu().numpy())

    all_ood_input = torch.cat(all_ood_input, 0)
    N = all_ood_input.shape[0]

    # non-sample version
    selected_indices = np.random.choice(N, ood_dataset_size, replace = False)
    selected_ood_conf = np.array(all_ood_conf)[selected_indices]
    softmax_selected_ood_conf = np.array(softmax_all_ood_conf)[selected_indices]
    print('Total OOD samples: ', len(selected_indices))
    print('Max OOD Conf: ', np.max(all_ood_conf), 'Min OOD Conf: ', np.min(all_ood_conf), 'Average OOD Conf: ', np.mean(all_ood_conf))
    print('Selected Max OOD Conf: ', np.max(selected_ood_conf), 'Selected Min OOD Conf: ', np.min(selected_ood_conf), 'Selected Average OOD Conf: ', np.mean(selected_ood_conf))
    print('(Softmax) Max OOD Conf: ', np.max(softmax_all_ood_conf), 'Min OOD Conf: ', np.min(softmax_all_ood_conf), 'Average OOD Conf: ', np.mean(softmax_all_ood_conf))
    print('(Softmax) Selected Max OOD Conf: ', np.max(softmax_selected_ood_conf), 'Selected Min OOD Conf: ', np.min(softmax_selected_ood_conf), 'Selected Average OOD Conf: ', np.mean(softmax_selected_ood_conf))
    ood_images = all_ood_input[selected_indices]
    ood_labels = (torch.ones(ood_dataset_size) * num_classes).long()

    ood_train_loader = torch.utils.data.DataLoader(
        OODDataset(ood_images, ood_labels),
        batch_size=batch_size, shuffle=True)

    print('Time: ', time.time()-start)

    return ood_train_loader

class OODDataset(torch.utils.data.Dataset):
  def __init__(self, images, labels):
        self.labels = labels
        self.images = images

  def __len__(self):
        return len(self.images)

  def __getitem__(self, index):
        # Load data and get label
        X = self.images[index]
        y = self.labels[index]

        return X, y

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

def save_checkpoint(state, epoch):
    """Saves checkpoint to disk"""
    directory = "checkpoints/{in_dataset}/{name}/".format(in_dataset=args.in_dataset, name=args.name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + 'checkpoint_{}.pth.tar'.format(epoch)
    torch.save(state, filename)


if __name__ == '__main__':
    main()