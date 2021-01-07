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
# training schedule
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--epochs', default= 30, type=int,
                    help='number of total epochs to run')
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
parser.add_argument('--ood_factor', type=float, default= 3,
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
parser.add_argument('--name', default = "debug_30_energy", type=str,
                    help='name of experiment')
parser.add_argument('--tensorboard',
                    help='Log progress to TensorBoard', action='store_true')
parser.add_argument('--log_name',
                    help='Name of the Log File', type = str, default = "info.log")
#Device options
parser.add_argument('--gpu-ids', default='0,1,2,3', type=str,
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
    label_transform = transforms.Compose([ToLabel()])

    if args.in_dataset == "pascal":
        train_set = pascalVOCSet("./datasets/pascal/", split="voc12-train", img_transform = img_transform, 
                                 label_transform = label_transform)
        val_set = pascalVOCSet('./datasets/pascal/', split="voc12-val",
                                   img_transform=val_transform, label_transform=label_transform)      
        # train_loaders = [torch.utils.data.DataLoader(sub_dataset, batch_size=args.batch_size, shuffle=True) 
        #                         for sub_dataset in torch.utils.data.random_split( train_set, [2858, 2859])]
        num_classes = 20
        pool_size = args.pool_size
        lr_schedule=[50, 75, 90]

        # normalizer = None
    elif args.in_dataset == "coco":
        # train_set = cocoloader("./datasets/coco_no_less_than_200/",
        #                     img_transform = img_transform,
        #                     label_transform = label_transform)
        # val_set = cocoloader("./datasets/coco_no_less_than_200/", split="multi-label-val2014",
        #                     img_transform = val_transform,
        #                     label_transform = label_transform)
        # num_classes = 80

        train_set = cocoloader("./datasets/coco_relabeled_67/",
                            img_transform = img_transform,
                            label_transform = label_transform)
        val_set = cocoloader("./datasets/coco_relabeled_67/", split="multi-label-val2014",
                            img_transform = val_transform,
                            label_transform = label_transform)
        num_classes = 67




        pool_size = args.pool_size
        lr_schedule=[50, 75, 90]

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, num_workers= 4, shuffle=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, num_workers= 4, shuffle=False, pin_memory=True)

    ood_dataset_size = int(len(train_loader.dataset) * args.ood_factor)
    print('OOD Dataset Size (per epoch): ', ood_dataset_size)

    ood_root = "/nobackup-slow/dataset/ILSVRC-2012/ood_train"
    ood_data = datasets.ImageFolder(ood_root, transform=transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]))
    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.ood_batch_size, num_workers= 4, shuffle= True, pin_memory=True)
    print('OOD Pool Size: ', len(ood_loader.dataset))
    # create model
    if args.model_arch == 'densenet':
        # model = dn.DenseNet3(args.layers, num_classes, args.growth, reduction=args.reduce,
        #                      bottleneck=args.bottleneck, dropRate=args.droprate, normalizer=normalizer)

        orig_densenet = torchvision.models.densenet121(pretrained=True)
        features = list(orig_densenet.features)
        model = nn.Sequential(*features, nn.ReLU(inplace=True))
        clsfier = clssimp(1024, num_classes)
        output_dim = 1
        repr_dim = 1000
        # model = torchvision.models.densenet121(pretrained=False)
    elif args.model_arch == "wideresnet50":
        orig_resnet = torchvision.models.wide_resnet50_2(pretrained=True)
        features = list(orig_resnet.children())
        model = nn.Sequential(*features[0:8])
        clsfier = clssimp(2048, num_classes)
        output_dim = 1
        repr_dim = 1000
    elif args.model_arch == "resnet101":
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

    else:
        assert False, 'Not supported model arch: {}'.format(args.model_arch)


    model = model.cuda()
    clsfier = clsfier.cuda()
    bayes_nn = NeuralLinear(args, model, clsfier, repr_dim, output_dim, num_classes)

    cudnn.benchmark = True

    # define loss function (criterion) and pptimizer
    # class_weights = torch.tensor([33.85365854, 39.68327402, 27.72361809, 42.3030303 , 27.65162907,
    #                              51.20091324, 17.40901771, 20.17037037, 16.42682927, 72.75483871,
    #                              34.94968553, 16.97484277, 47.03361345, 40.72262774,  4.33706816,
    #                              38.55709343, 65.85380117, 30.84401114, 40.57090909, 37.23411371, 1.0])
    # class_weights = torch.tensor( [1]*20 + [500])
    # criterion_in = nn.BCEWithLogitsLoss(pos_weight = class_weights).cuda() # sigmoid 
    criterion = nn.BCEWithLogitsLoss().cuda() # sigmoid 

    optimizer = torch.optim.Adam([{'params': model.parameters(),'lr':args.lr/10},
                                {'params': clsfier.parameters()}], lr=args.lr)

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
            # print(m)
            m.eval()
            if freeze_bn_affine:
                m.weight.requires_grad = False
                m.bias.requires_grad = False
    #CORE
    best_map = 0.0
    mAPs = []
    # bayes_nn.sample_BDQN()
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, lr_schedule)
        print("in every epoch, the total ood pool size is : ", args.ood_batch_size * args.pool_size )
        print("will sample {} ood from it for training".format(ood_dataset_size)) 
        selected_ood_loader = select_ood(ood_loader, model, clsfier, bayes_nn, args.batch_size * args.ood_factor, 
                                            num_classes, pool_size, ood_dataset_size, log)
        torch.cuda.empty_cache()                                    
        bayes_nn.train_blr_energy(train_loader, selected_ood_loader, criterion, optimizer, epoch, directory, log)
       # bayes_nn.update_representation()
       # bayes_nn.update_bays_reg_BDQN(log)
       # bayes_nn.sample_BDQN()
    
        # evaluate on validation set
        mAP = bayes_nn.validate_energy(val_loader, epoch, log)
        if mAP > best_map:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict_model': model.state_dict(),
                'state_dict_clsfier': clsfier.state_dict(),
            }, epoch + 1)
            best_map = mAP
            mAPs.append(mAP)
            log.debug("Epoch [%d/%d][saved] mAP: %.4f" % (epoch + 1, args.epochs, mAP))

            # test_mAP = validate(args, model, clsfier, test_loader, num_classes, True)
            # print("Epoch [%d/%d][test set] test_mAP: %.4f" % (epoch + 1, args.epochs, test_mAP))
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
            # print("Epoch [%d/%d][----] mAP: %.4f" % (epoch + 1, args.epochs, mAP))

    torch.save(torch.tensor(mAPs), os.path.join(directory, "all_mAPs.data") )     


def select_ood(ood_loader, model, clsfier, bayes_model, batch_size, num_classes, pool_size, ood_dataset_size, log):

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
    clsfier.eval()
    with torch.no_grad():
        all_ood_input = []
        # clsfier_all_ood_conf = []
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

            input = out_set[0] 
            #Clsifier branch results
            # model.to(input.device)
            # clsfier_output = clsfier(model(input.cuda()))
            # clsfier_prob = torch.sigmoid(clsfier_output)
            # clsfier_all_ood_conf.extend(clsfier_prob[:,-1].unsqueeze(-1).detach().cpu().numpy())

            all_ood_input.append(input)

        all_ood_input = torch.cat(all_ood_input, 0)
        N = all_ood_input.shape[0]
        print(f"N is {N}")
        # clsfier_all_ood_conf = np.array(clsfier_all_ood_conf).squeeze()

        selected_indices = np.random.choice(N, ood_dataset_size, replace = False)
        # selected_indices = argmin_abs_val[: ood_dataset_size]
        # clsfier_selected_ood_conf = clsfier_all_ood_conf[selected_indices]
        print('Total OOD samples: ', len(selected_indices))
        # log.debug(f'(Softmax classifier) Max OOD Conf: {np.max(clsfier_all_ood_conf)} Min OOD Conf: {np.min(clsfier_all_ood_conf)} Average OOD Conf: {np.mean(clsfier_all_ood_conf)}')
        # log.debug(f'(Softmax classifier) Selected Max OOD Conf: {np.max(clsfier_selected_ood_conf)} Selected Min OOD Conf: {np.min(clsfier_selected_ood_conf)} Selected Average OOD Conf: {np.mean(clsfier_selected_ood_conf)}')
        ood_images = all_ood_input[selected_indices]
        ood_labels = (torch.ones(ood_dataset_size) * -1).long()

        ood_train_loader = torch.utils.data.DataLoader(
            SimpleDataset(ood_images, ood_labels),
            batch_size=batch_size, shuffle=True)

    print('Time: ', time.time()-start)

    return ood_train_loader



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




if __name__ == '__main__':
    main()