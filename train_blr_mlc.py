import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
# os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
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
from neural_linear_mlc import NeuralLinear
from utils import LinfPGDAttack, TinyImages
# used for logging to TensorBoard
from tensorboard_logger import configure, log_value
from torch.distributions.multivariate_normal import MultivariateNormal

import json
import logging

from torch.utils.data import Sampler
from utils.pascal_voc_loader import pascalVOCSet
from utils import cocoloader
from utils.transform import ReLabel, ToLabel, ToSP, Scale
from utils import ImageNet
from sklearn import metrics
from neural_linear_mlc import SimpleDataset

parser = argparse.ArgumentParser(description='OOD training for multi-label classification')
# parser.add_argument('--gpu', default=3, type=int, help='the preferred gpu to use')

parser.add_argument('--in-dataset', default="pascal", type=str, help='in-distribution dataset e.g. pascal')
parser.add_argument('--auxiliary-dataset', default='imagenet', 
                    choices=['80m_tiny_images', 'imagenet'], type=str, help='which auxiliary dataset to use')
parser.add_argument('--model-arch', default='densenet', type=str, help='model architecture e.g. simplenet densenet')

parser.add_argument('--save-epoch', default= 10, type=int,
                    help='save the model every save_epoch') # freq; save model state_dict()
parser.add_argument('--save-data-epoch', default= 120, type=int,
                    help='save the sampled ood every save_data_epoch') # freq; save model state_dict()
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    help='print frequency (default: 10)') # print every print-freq batches during training

# ID train & val batch size and OOD train batch size 
parser.add_argument('-b', '--batch-size', default= 64, type=int,
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
parser.add_argument('--beta', default=1.0, type=float, help='beta for out_loss')


# ood sampling and mining
parser.add_argument('--ood-batch-size', default= 100, type=int,
                    help='mini-batch size (default: 400) used for ood mining')
parser.add_argument('--pool-size', default= 200, type=int,
                    help='pool size')
parser.add_argument('--ood_factor', type=float, default= 2,
                 help='ood_dataset_size = len(train_loader.dataset) * ood_factor default = 2.0')

#posterior sampling
parser.add_argument('--a0', type=float, default=6.0, help='a0')
parser.add_argument('--b0', type=float, default=6.0, help='b0')
parser.add_argument('--lambda_prior', type=float, default=0.25, help='lambda_prior')
parser.add_argument('--sigma', type=float, default=100, help='control var for weights')
parser.add_argument('--sigma_n', type=float, default=1, help='control var for noise')
parser.add_argument('--conf', type=float, default=4.6, help='control ground truth for bayesian linear regression.3.9--0.98; 4.6 --0.99')
# saving, naming and logging
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--name', default = "debug_test_hard_400 ", type=str,
                    help='name of experiment')
parser.add_argument('--tensorboard',
                    help='Log progress to TensorBoard', action='store_true')
parser.add_argument('--log_name',
                    help='Name of the Log File', type = str, default = "info.log")

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



torch.manual_seed(1)
np.random.seed(1)

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

    if args.in_dataset == "pascal":
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        label_transform = transforms.Compose([
                ToLabel(),
            ])
        train_set = pascalVOCSet("./datasets/pascal/", split="voc12-train", img_transform = transforms.Compose([
                                            transforms.RandomHorizontalFlip(),
                                            transforms.RandomResizedCrop((256),scale=(0.5, 2.0)),
                                            transforms.ToTensor(),
                                            normalize]), 
                                 label_transform = label_transform)
        val_set = pascalVOCSet('./datasets/pascal/', split="voc12-val",
                                   img_transform=transforms.Compose([
                                    transforms.Resize((256, 256)),
                                    transforms.ToTensor(),
                                    normalize]),
                                   label_transform=label_transform)
        test_set = pascalVOCSet('./datasets/pascal/', split="voc12-test",
                                    img_transform=transforms.Compose([
                                    transforms.Resize((256, 256)),
                                    transforms.ToTensor(),
                                    normalize]), label_transform = None)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, num_workers=1, shuffle=True, pin_memory=True)
        train_loaders = [torch.utils.data.DataLoader(sub_dataset, batch_size=args.batch_size, shuffle=True) 
                                for sub_dataset in torch.utils.data.random_split( train_set, [2858, 2859])]
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, num_workers=1, shuffle=False, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, num_workers=1, shuffle=False, pin_memory=True)

        # lr_schedule=[50, 75, 90]
        num_classes = 20
        pool_size = args.pool_size
        lr_schedule=[50, 75, 90]

        # normalizer = None
    elif args.dataset == "coco":
        train_set = cocoloader("./datasets/coco/",
                            img_transform = img_transform,
                            label_transform = label_transform)
        val_set = cocoloader("./datasets/coco/", split="multi-label-val2014",
                            img_transform = val_transform,
                            label_transform = label_transform)

    ood_dataset_size = int(len(train_loader.dataset) * args.ood_factor)
    print('OOD Dataset Size: ', ood_dataset_size)

    ood_root = "/nobackup-slow/dataset/ILSVRC-2012/train"
    ood_data = datasets.ImageFolder(ood_root, transform=transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        normalize,
    ]))
    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.ood_batch_size, num_workers=1, shuffle= True, pin_memory=True)

    # create model
    if args.model_arch == 'densenet':
        # model = dn.DenseNet3(args.layers, num_classes, args.growth, reduction=args.reduce,
        #                      bottleneck=args.bottleneck, dropRate=args.droprate, normalizer=normalizer)

        orig_densenet = torchvision.models.densenet121(pretrained=True)
        features = list(orig_densenet.features)
        model = nn.Sequential(*features, nn.ReLU(inplace=True))
        clsfier = clssimp(1024, num_classes + 1)
        output_dim = 1
        repr_dim = 1000
        # model = torchvision.models.densenet121(pretrained=False)
    else:
        assert False, 'Not supported model arch: {}'.format(args.model_arch)


    model = model.cuda()
    clsfier = clsfier.cuda()
    bayes_nn = NeuralLinear(args, model, clsfier, repr_dim, output_dim)

    cudnn.benchmark = True

    # define loss function (criterion) and pptimizer
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
            m.eval()
            if freeze_bn_affine:
                m.weight.requires_grad = False
                m.bias.requires_grad = False
    #CORE
    best_map = 0.0
    mAPs = []
    bayes_nn.sample_BDQN()
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, lr_schedule)
        print("in every epoch, the total ood pool size is : ", args.ood_batch_size * args.pool_size )
        print("will sample {} ood from it for training".format(ood_dataset_size)) 
        # for sub_train_loader in train_loaders: 
        #     selected_ood_loader = select_ood(ood_loader, model, clsfier, bayes_nn, args.batch_size * 2, 
        #                                      num_classes, pool_size, ood_dataset_size)
        #     bayes_nn.train_blr(sub_train_loader, selected_ood_loader, model, clsfier, criterion, num_classes, optimizer, epoch, directory)
        #     bayes_nn.update_representation()
        #     bayes_nn.update_bays_reg_BDQN()
        #     bayes_nn.sample_BDQN()
        selected_ood_loader = select_ood(ood_loader, model, clsfier, bayes_nn, args.batch_size * 2, 
                                            num_classes, pool_size, ood_dataset_size, log)
        torch.cuda.empty_cache()                                    
        bayes_nn.train_blr(train_loader, selected_ood_loader, model, clsfier, criterion, num_classes, optimizer, epoch, directory, log)
        bayes_nn.update_representation()
        bayes_nn.update_bays_reg_BDQN(log)
        bayes_nn.sample_BDQN()
    
        # evaluate on validation set
        mAP = validate(args, model, clsfier, val_loader, num_classes, epoch)
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
            if epoch % 2 == 0:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict_model': model.state_dict(),
                    'state_dict_clsfier': clsfier.state_dict(),
                }, epoch + 1)
            best_map = mAP
            mAPs.append(mAP)
            log.debug("Epoch [%d/%d][saved] mAP: %.4f" % (epoch + 1, args.epochs, mAP))
            # print("Epoch [%d/%d][----] mAP: %.4f" % (epoch + 1, args.epochs, mAP))
                
        #DEBUG
        log.debug("cov and mean checking...")
        log.debug("cov:", bayes_nn.cov_w[0,:6,:6])
        
        log.debug("mu: ", bayes_nn.mu_w[0][:20])

    torch.save(torch.tensor(mAPs), os.path.join(directory, "all_mAPs.data") )     


def validate(args, model, clsfier, val_loader, num_classes, epoch):
    model.eval()
    clsfier.eval()
    init = True
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
            outputs = torch.sigmoid(outputs)
            pred = outputs[:, :-1].squeeze().data.cpu().numpy()
            ground_truth = labels.squeeze().data.cpu().numpy()
            if init:
                 all_ground_truth = ground_truth 
                 all_pred = pred
                 init = False
            else:
                all_ground_truth = np.vstack((all_ground_truth, ground_truth) )
                all_pred = np.vstack((all_pred, pred))
            # for label in range(0, num_classes):
            #     gts[label].append(ground_truth[label])
            #     preds[label].append(pred[label])

    FinalMAPs = []
    for i in range(0, num_classes):
        precision, recall, thresholds = metrics.precision_recall_curve(all_ground_truth[i], all_pred[i])
        FinalMAPs.append(metrics.auc(recall, precision))
    # print(FinalMAPs)
    print(np.mean(FinalMAPs))
    
    # # log to TensorBoard
    # if self.args.tensorboard:
    #     log_value('mAP', np.mean(FinalMAPs) epoch)

    return np.mean(FinalMAPs)

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
        all_abs_val = []
        all_ood_conf = []
        clsfier_all_ood_conf = []
        for k in range(pool_size): 
            if k % 10 == 0:
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
            output = bayes_model.predict(input.cuda())
            abs_val = torch.abs(output).squeeze() 
            prob = torch.sigmoid(output)  
            
            #Clsifier branch results
            # model.to(input.device)
            clsfier_output = clsfier(model(input.cuda()))
            clsfier_prob = torch.sigmoid(clsfier_output)
            clsfier_all_ood_conf.extend(clsfier_prob[:,-1].unsqueeze(-1).detach().cpu().numpy())

            all_ood_input.append(input)
            all_ood_conf.extend(prob.detach().cpu().numpy())
            all_abs_val.extend(abs_val.detach().cpu().numpy())
            

        all_ood_input = torch.cat(all_ood_input, 0)
        all_abs_val = np.array(all_abs_val)
        all_ood_conf = np.array(all_ood_conf)
        argmin_abs_val = np.argsort(all_abs_val) 
        N = all_ood_input.shape[0]
        clsfier_all_ood_conf = np.array(clsfier_all_ood_conf).squeeze()

        # non-sample version
        selected_indices = argmin_abs_val[: ood_dataset_size]
        selected_ood_conf = all_ood_conf[selected_indices]
        clsfier_selected_ood_conf = clsfier_all_ood_conf[selected_indices]
        print('Total OOD samples: ', len(selected_indices))
        log.debug(f'Max OOD Conf: {np.max(all_ood_conf)} Min OOD Conf: {np.min(all_ood_conf)} Average OOD Conf: {np.mean(all_ood_conf)}')
        log.debug(f'Selected Max OOD Conf: {np.max(selected_ood_conf)} Selected Min OOD Conf: {np.min(selected_ood_conf)} Selected Average OOD Conf: {np.mean(selected_ood_conf)}')
        log.debug(f'(Softmax classifier) Max OOD Conf: {np.max(clsfier_all_ood_conf)} Min OOD Conf: {np.min(clsfier_all_ood_conf)} Average OOD Conf: {np.mean(clsfier_all_ood_conf)}')
        log.debug(f'(Softmax classifier) Selected Max OOD Conf: {np.max(clsfier_selected_ood_conf)} Selected Min OOD Conf: {np.min(clsfier_selected_ood_conf)} Selected Average OOD Conf: {np.mean(clsfier_selected_ood_conf)}')
        ood_images = all_ood_input[selected_indices]
        ood_labels = (torch.ones(ood_dataset_size) * num_classes).long()

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


class clssimp(nn.Module):
    def __init__(self, ch=2880, num_classes=20):

        super(clssimp, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.way1 = nn.Sequential(
            nn.Linear(ch, 1000, bias=True),
            nn.BatchNorm1d(1000),
            nn.ReLU(inplace=True),
        )

        self.cls= nn.Linear(1000, num_classes,bias=True)

    def forward(self, x):
        # bp()
        x = self.pool(x)
        x = x.reshape(x.size(0), -1)
        x = self.way1(x)
        logits = self.cls(x)
        return logits

    def intermediate_forward(self, x):
        x = self.pool(x)
        x = x.reshape(x.size(0), -1)
        x = self.way1(x)
        return x


if __name__ == '__main__':
    main()