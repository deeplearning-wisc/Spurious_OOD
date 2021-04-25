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

# import utils.svhn_loader as svhn
from models.fine_tuning_layer import clssimp as clssimp
import models.densenet as dn
import models.wideresnet as wn
import models.resnet as rn
import models.simplenet as sn
# from neural_linear_mlc import NeuralLinear
# from utils import LinfPGDAttack, TinyImages
# used for logging to TensorBoard
from tensorboard_logger import configure, log_value
from torch.distributions.multivariate_normal import MultivariateNormal
from rebias_utils import SimpleConvNet
from dann_utils import CNNModel

from torch.utils.data import Sampler
# from utils.pascal_voc_loader import pascalVOCSet
# from utils import cocoloader
# from utils.transform import ReLabel, ToLabel, ToSP, Scale
# from utils import ImageNet
# from neural_linear_mlc import SimpleDataset
from datasets.color_mnist import get_biased_mnist_dataloader
import cv2
from torch.utils.data.dataloader import default_collate

parser = argparse.ArgumentParser(description='OOD Detection Evaluation based on Energy-score')

parser.add_argument('--in-dataset', default="color_mnist", type=str, help='in-distribution dataset')
parser.add_argument('--model-arch', default='resnet18', type=str, help='model architecture')
parser.add_argument('--print-freq', '-p', default=10, type=int, help='print frequency (default: 10)') # print every print-freq batches during training
# ID train & val batch size and OOD train batch size 
parser.add_argument('-b', '--batch-size', default= 64, type=int,
                    help='mini-batch size (default: 64) used for training id and ood')
# # densenet
# parser.add_argument('--layers', default= 100, type=int,
#                     help='total number of layers (default: 100) for DenseNet')
# parser.add_argument('--growth', default= 12, type=int,
#                     help='number of new channels per layer (default: 12)')
# # network spec
# parser.add_argument('--droprate', default=0.0, type=float,
#                     help='dropout probability (default: 0.0)')
# parser.add_argument('--no-augment', dest='augment', action='store_false',
#                     help='whether to use standard augmentation (default: True)')
# parser.add_argument('--reduce', default=0.5, type=float,
#                     help='compression rate in transition stage (default: 0.5)')
# parser.add_argument('--no-bottleneck', dest='bottleneck', action='store_false',
#                     help='To not use bottleneck block')
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

parser.set_defaults(bottleneck=True)
parser.set_defaults(augment=True)

args = parser.parse_args()

state = {k: v for k, v in args._get_kwargs()}

print(state)
directory = "checkpoints/{in_dataset}/{name}/".format(in_dataset=args.in_dataset, name=args.name)
# if not os.path.exists(directory):
#     os.makedirs(directory)
# save_state_file = os.path.join(directory, 'args.txt')
# fw = open(save_state_file, 'w')
# print(state, file=fw)
# fw.close()

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
torch.manual_seed(1)
np.random.seed(1)
np.random.seed(1)
use_cuda = torch.cuda.is_available()
if use_cuda:
    torch.cuda.manual_seed_all(1)

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
    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    # normalizer = transforms.Normalize((125.3/255, 123.0/255, 113.9/255), (63.0/255, 62.1/255.0, 66.7/255.0)) #obsolete
    val_transform = transforms.Compose([transforms.Scale(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        normalizer])
    if args.in_dataset == "CIFAR-10":
        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./datasets/cifar10', train=False, transform=transform_test),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        num_classes = 10
    elif args.in_dataset == "CIFAR-100":
        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('./datasets/cifar100', train=False, transform=transform_test),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        num_classes = 100
    elif args.in_dataset == "IN-9":
        # test_set = torchvision.datasets.ImageFolder("/nobackup-slow/dataset/background/mixed_rand/val",
        #                                 transform= val_transform)
        # val_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size,
        #                                      shuffle=True, num_workers=4)
        test_set = torchvision.datasets.ImageFolder(root="/nobackup-slow/dataset/background/original/val", 
                  transform=val_transform)
        val_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size,
                                         shuffle=True, num_workers=4)
        val_loader_cam = torch.utils.data.DataLoader(test_set, batch_size=1,
                                         shuffle=True, num_workers=2)
        num_classes = 9
    elif args.in_dataset == "random":
        test_set = torchvision.datasets.ImageFolder(root="/nobackup-slow/dataset/shape/val", transform=val_transform)  
        # test_set = torchvision.datasets.ImageFolder(root="datasets/random_shape/val_4", transform=val_transform)
        val_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size,
                                         shuffle=True, num_workers=4)
        val_loader_cam = torch.utils.data.DataLoader(test_set, batch_size=1,
                                         shuffle=True, num_workers=2)
        num_classes = 9

    elif args.in_dataset == "color_mnist":
        # val_loader  = get_ood_loader("0_background")
        val_loader = get_biased_mnist_dataloader(root = './datasets/MNIST', batch_size=args.batch_size,
                                            data_label_correlation= args.data_label_correlation,
                                            n_confusing_labels= 4,
                                            train=False, partial=True, cmap = "2")
        val_loader_cam = get_biased_mnist_dataloader(root = './datasets/MNIST', batch_size=1,
                                            data_label_correlation= args.data_label_correlation,
                                            n_confusing_labels= 9,
                                            train=False, partial=False, cmap = "2")
        num_classes = 5
    # create model
    # if args.model_arch == 'densenet':
    #     model = dn.DenseNet3(args.layers, num_classes, normalizer=normalizer)
    # elif args.model_arch == 'wideresnet':
    #     model = wn.WideResNet(args.depth, num_classes, widen_factor=args.width, dropRate=args.droprate, normalizer=normalizer)
    if args.model_arch == "resnet101":
        orig_resnet = rn.l_resnet101()
        features = list(orig_resnet.children())
        model = nn.Sequential(*features[0:8])
        clsfier = clssimp(2048, num_classes)
    elif args.model_arch == "resnet18":
        orig_resnet = torchvision.models.resnet18(pretrained=False)
        features = list(orig_resnet.children())
        model = nn.Sequential(*features[0:8])
        clsfier = clssimp(512, num_classes)
    elif args.model_arch == "rebias_conv":
        f_config = {'num_classes': 5, 'kernel_size': 7, 'feature_pos': 'post'}
        model = SimpleConvNet(**f_config).cuda()
        clsfier = None
    elif args.model_arch == "dann":
        model = CNNModel(num_classes=num_classes).cuda()
        clsfier = None
    else:
        assert False, 'Not supported model arch: {}'.format(args.model_arch)

    test_epochs = args.test_epochs.split()
    if args.in_dataset == 'IN-9':
        # out_datasets = ['places365','LSUN', 'LSUN_resize', 'iSUN', 'dtd', 'SVHN',"ocean", 'fox']
        # out_datasets = ["ocean", 'fox']
        # out_datasets = ['no_fg','only_bg_b', 'places365', 'SVHN', 'ocean']
        out_datasets = ['no_fg','only_bg_b', 'ocean']
    elif args.in_dataset == 'random':
        out_datasets = ['red_rectangle', 'green_rectangle']
    elif args.in_dataset == 'color_mnist':
         # out_datasets = ['partial_color_mnist_0&1']
         out_datasets = ['partial_color_mnist']
    # load model and store test results


    for test_epoch in test_epochs:
        checkpoint = torch.load("./checkpoints/{in_dataset}/{name}/checkpoint_{epochs}.pth.tar".format(in_dataset=args.in_dataset, name=args.name, epochs= test_epoch))
        # model.load_state_dict(checkpoint['state_dict'])
        model.load_state_dict(checkpoint['state_dict_model'])
        model.eval()
        model.cuda()
        if args.model_arch != "rebias_conv" and args.model_arch != "dann":
            clsfier.load_state_dict(checkpoint['state_dict_clsfier'])
            clsfier.eval()
            clsfier.cuda()
        save_dir =  f"./energy_results/{args.in_dataset}/{args.name}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        print("processing ID")
        # id_sum_energy, id_cmt = get_energy(args, model, clsfier, val_loader, test_epoch, log, id = False)
        # id_sum_energy = get_energy_biased(args, model, clsfier, val_loader, test_epoch, log, id = False)
        id_sum_energy,id_cmt = get_energy_biased(args, model, clsfier, val_loader, test_epoch, log, model_arch=args.model_arch, id = True)
        with open(os.path.join(save_dir, f'energy_score_at_epoch_{test_epoch}.npy'), 'wb') as f:
            np.save(f, id_sum_energy)
            # np.save(f, id_cmt)
        for out_dataset in out_datasets:
            print("processing OOD dataset ", out_dataset)
            testloaderOut = get_ood_loader(out_dataset)
            ood_sum_energy = get_energy(args, model, clsfier, testloaderOut, test_epoch, log, model_arch=args.model_arch, id = False)
            # ood_sum_energy, ood_cmt = get_energy(args, model, clsfier, testloaderOut, test_epoch, log, id = True)
            with open(os.path.join(save_dir, f'energy_score_{out_dataset}_at_epoch_{test_epoch}.npy'), 'wb') as f:
                np.save(f, ood_sum_energy)
                # np.save(f, ood_cmt)

    if args.CAM:
        checkpoint = torch.load("./checkpoints/{in_dataset}/{name}/checkpoint_{epochs}.pth.tar".format(in_dataset=args.in_dataset, name=args.name, epochs= test_epochs[0]))
        # model.load_state_dict(checkpoint['state_dict'])
        model.load_state_dict(checkpoint['state_dict_model'])
        clsfier.load_state_dict(checkpoint['state_dict_clsfier'])
        model.eval()
        model.cuda()
        clsfier.eval()
        clsfier.cuda()
        # get_cam(args, model, clsfier, val_loader_cam, args.in_dataset, num_result = 20)
        get_cam_per_class(args, model, clsfier, val_loader_cam, args.in_dataset, num_result = 120, size_upsample = (32, 32))
        # out_dataset = out_datasets[0]
        # testloaderOut, testloaderOut_cam = get_ood_loader(out_dataset, CAM  = True)
        # get_cam_per_class(args, model, clsfier,  testloaderOut_cam, out_dataset, num_result = 120, size_upsample = (32, 32))

def get_energy(args, model, clsfier, val_loader, epoch, log, model_arch, id = False, CAM = False):
    in_energy = AverageMeter()
    top1 = AverageMeter()
    model.eval()
    if clsfier != None:
        clsfier.eval()
    init = True
    log.debug("######## Start collecting energy score ########")
    with torch.no_grad():
        if id:
            all_preds = torch.tensor([])
            all_targets = torch.tensor([])
        for i, (images, labels) in enumerate(val_loader):
            images = images.cuda()
            if model_arch == "rebias_conv":
                outputs, _ = model(images)
            elif model_arch == "dann":
                outputs, _ = model(images, alpha=0)
            else:
                outputs = model(images)
                outputs = clsfier(outputs)
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
            print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
            stacked = torch.stack((all_targets,all_preds),dim=1)
            cmt = torch.zeros(9,9, dtype=torch.int64)
            for p in stacked:
                tl, pl = p.type(torch.int64).tolist()
                cmt[tl, pl] = cmt[tl, pl] + 1
            return sum_energy, cmt
        else:
            return sum_energy

def get_energy_biased(args, model, clsfier, val_loader, epoch, log, model_arch, id = False, CAM = False):
    in_energy = AverageMeter()
    top1 = AverageMeter()
    model.eval()
    if clsfier != None:
        clsfier.eval()
    init = True
    log.debug("######## Start collecting energy score ########")
    with torch.no_grad():
        if id:
            all_preds = torch.tensor([])
            all_targets = torch.tensor([])
        for i, (images, labels, _) in enumerate(val_loader):
            images = images.cuda()
            if model_arch == "rebias_conv":
                outputs, _ = model(images)
            elif model_arch == "dann":
                outputs, _ = model(images, alpha=0)
            else:
                outputs = model(images)
                outputs = clsfier(outputs)
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
            print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
            stacked = torch.stack((all_targets,all_preds),dim=1)
            cmt = torch.zeros(5,5, dtype=torch.int64)
            for p in stacked:
                tl, pl = p.type(torch.int64).tolist()
                cmt[tl, pl] = cmt[tl, pl] + 1
            return sum_energy, cmt
        else:
            return sum_energy


def get_cam(args, model, clsfier, val_loader, dataset, num_result = 10, size_upsample = (224, 224)):
    
    def returnCAM(feature_conv, weight_softmax, class_idx, size_upsample):
        size_upsample = size_upsample
        _, nc, h, w = feature_conv.shape
        output_cam = []
        # cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h*w)))
        cam = weight_softmax[class_idx] * (feature_conv[0][class_idx])
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
        return output_cam

    finalconv_name = 'conv'
    result_path = f"./cam_results/{args.name}/{dataset}"
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    # hook
    feature_blobs = []
    def hook_feature(module, input, output):
        feature_blobs.append(output.cpu().data.numpy())
    clsfier._modules.get(finalconv_name).register_forward_hook(hook_feature)
    params = list(clsfier.parameters())
    # get weight only from the last layer(linear)
    weight_softmax = np.squeeze(params[-1].cpu().data.numpy())
    print("######## Start generating CAM ########")
    with torch.no_grad():
        for i, (images, labels, _) in enumerate(val_loader):
            # unorm = UnNormalize(mean=(125.3/255, 123.0/255, 113.9/255), std=(63.0/255, 62.1/255.0, 66.7/255.0)) #obsolete
            unorm = UnNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            unorm = UnNormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

            images = images.cuda()
            targets = labels.cuda()
            outputs = model(images)
            outputs = clsfier(outputs)
            h_x = F.softmax(outputs, dim=1).data.squeeze()
            probs, idx = h_x.sort(0, True)
            print("%d True label : %d, Predicted label : %d, Probability : %.2f" % (i+1, labels.item(), idx[0].item(), probs[0].item()))
            CAMs = returnCAM(feature_blobs[0], weight_softmax, [idx[0].item()], size_upsample =  size_upsample )
            unorm(images[0])
            image_PIL = transforms.ToPILImage()(images[0])
            image_PIL.save(os.path.join(result_path, 'img%d.png' % (i + 1)))
            
            img = cv2.imread(os.path.join(result_path, 'img%d.png' % (i + 1)))
            height, width, _ = img.shape
            heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (width, height)), cv2.COLORMAP_JET)
            result = heatmap * 0.5 + img * 0.3
            cv2.imwrite(os.path.join(result_path, 'cam%d.png' % (i + 1)), result)
            if i + 1 == num_result:
                break
            feature_blobs.clear()

def get_cam_per_class(args, model, clsfier, val_loader, dataset, num_result = 10, size_upsample = (224, 224)):
    idx_to_class = {v: k for k, v in val_loader.dataset.class_to_idx.items()}
    def returnCAM(feature_conv, weight_softmax, class_idx, size_upsample ):
        size_upsample = size_upsample 
        _, nc, h, w = feature_conv.shape
        output_cam = []
        # cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h*w)))
        cam = weight_softmax[class_idx] * (feature_conv[0][class_idx])
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
        return output_cam

    finalconv_name = 'conv'
    result_path = f"./cam_results/{args.name}/{dataset}"
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    # hook
    feature_blobs = []
    def hook_feature(module, input, output):
        feature_blobs.append(output.cpu().data.numpy())
    clsfier._modules.get(finalconv_name).register_forward_hook(hook_feature)
    params = list(clsfier.parameters())
    # get weight only from the last layer(linear)
    weight_softmax = np.squeeze(params[-1].cpu().data.numpy())
    print("######## Start generating CAM ########")
    with torch.no_grad():
        for i, (images, labels, _) in enumerate(val_loader):
            # unorm = UnNormalize(mean=(125.3/255, 123.0/255, 113.9/255), std=(63.0/255, 62.1/255.0, 66.7/255.0))
            unorm = UnNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            unorm = UnNormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

            images = images.cuda()
            targets = labels.cuda()
            outputs = model(images)
            outputs = clsfier(outputs)
            h_x = F.softmax(outputs, dim=1).data.squeeze()
            probs, idx = h_x.sort(0, True)
            print(f"{i+1} True label : {idx_to_class[labels.item()]} \t Predicted label : {idx_to_class[idx[0].item()]} \t Correct: {labels.item() == idx[0].item()} \t Prob : {probs[0].item():.2f}" )
            # print("%d True label : %d, Predicted label : %d, Probability : %.2f" % (i+1, labels.item(), idx[0].item(), probs[0].item()))
            CAMs = returnCAM(feature_blobs[0], weight_softmax, [idx[0].item()], size_upsample = size_upsample)
            unorm(images[0])
            image_PIL = transforms.ToPILImage()(images[0])

            class_path = os.path.join(result_path, idx_to_class[labels.item()])
            if not os.path.exists(class_path):
                os.makedirs(class_path)
            image_PIL.save(os.path.join(class_path, f'img{i+1}.png'))
            # img = cv2.imread(os.path.join(class_path, f'img{i+1}.png'))
            # image_PIL.save(os.path.join(class_path, f'img{i+1}_P{idx[0].item()}_T{labels.item()}.png'))
            # img = cv2.imread(os.path.join(class_path, f'img{i+1}_P{idx[0].item()}_T{labels.item()}.png'))
            # height, width, _ = img.shape
            # heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (width, height)), cv2.COLORMAP_JET)
            # result = heatmap * 0.5 + img * 0.3
            # cv2.imwrite(os.path.join(class_path, 'cam%d.png' % (i + 1)), result)
            if i + 1 == num_result:
                break
            feature_blobs.clear()

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
        normalizer = transforms.Normalize((125.3/255, 123.0/255, 113.9/255), (63.0/255, 62.1/255.0, 66.7/255.0))
        val_transform  =  transforms.Compose([transforms.Scale(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        normalizer])
        if out_dataset == 'SVHN':
            testsetout = svhn.SVHN('datasets/ood_datasets/svhn/', split='test',
                                  transform= val_transform, download=False)
            testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=args.ood_batch_size,
                                             shuffle=True, num_workers=2)
        # elif out_dataset == "CIFAR-100":
        #     testloaderOut = torch.utils.data.DataLoader(
        #         datasets.CIFAR100('./datasets/cifar100', train=False, download=False,
        #                         transform=transforms.Compose([transforms.Resize(32), transforms.CenterCrop(32), transforms.ToTensor()])
        #                         ),
        #     batch_size=args.ood_batch_size, shuffle=True, num_workers=2)
        # elif out_dataset == "CelebA":
        #     testsetout = torchvision.datasets.ImageFolder(root="/nobackup-slow/dataset/celeba",
        #                                 transform=transforms.Compose([transforms.Resize(32), transforms.CenterCrop(32), transforms.ToTensor()]))
        #     subset = torch.utils.data.Subset(testsetout, np.random.choice(len(testsetout), 10000, replace=False))
        #     testloaderOut = torch.utils.data.DataLoader(subset, batch_size=args.ood_batch_size, shuffle=True,
        #                                              num_workers=2)
        # elif out_dataset == 'dtd':
        #     testsetout = torchvision.datasets.ImageFolder(root="datasets/ood_datasets/dtd/images",
        #                                 transform=transforms.Compose([transforms.Resize(32), transforms.CenterCrop(32), transforms.ToTensor()]))
        #     testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=args.ood_batch_size, shuffle=True,
        #                                              num_workers=2)
        elif out_dataset == 'places365':
            testsetout = torchvision.datasets.ImageFolder(root="/nobackup-slow/dataset/places365",
            # testsetout = torchvision.datasets.ImageFolder(root="/nobackup-slow/dataset/places365/test_subset",
                transform=val_transform)
            subset = torch.utils.data.Subset(testsetout, np.random.choice(len(testsetout), 10000, replace=False))
            testloaderOut = torch.utils.data.DataLoader(subset, batch_size=args.ood_batch_size,
                                                     num_workers=2, shuffle=True)
        elif out_dataset == 'ocean' or out_dataset == 'coral':
            testsetout = torchvision.datasets.ImageFolder("./datasets/ood_datasets/{}".format(out_dataset),
                                        transform=val_transform)
            testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=args.ood_batch_size,
                                             shuffle=True, num_workers=2)
            testloaderOut_cam = torch.utils.data.DataLoader(testsetout, batch_size= 1,
                                             shuffle=True, num_workers=2)
        elif 'rectangle' in out_dataset:
            testsetout = torchvision.datasets.ImageFolder("./datasets/random_shape/ood/{}".format(out_dataset),
                                        transform=val_transform)
            testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=args.ood_batch_size,
                                             shuffle=True, num_workers=2)
            testloaderOut_cam = torch.utils.data.DataLoader(testsetout, batch_size= 1,
                                             shuffle=True, num_workers=2)
        elif 'partial_color_mnist' in out_dataset or out_dataset == "0_1_cross" or out_dataset == "0_background":
            val_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5),
                             std=(0.5, 0.5, 0.5))])
            testsetout = torchvision.datasets.ImageFolder("./datasets/ood_datasets/{}".format(out_dataset),
                                        transform=val_transform)
            testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=args.ood_batch_size,
                                             shuffle=True, num_workers=2)
            testloaderOut_cam = torch.utils.data.DataLoader(testsetout, batch_size= 1,
                                             shuffle=True, num_workers=2)

            # testloaderOut = get_biased_mnist_dataloader(root = './datasets/MNIST', batch_size=args.ood_batch_size,
            #                                 data_label_correlation= 1,
            #                                 n_confusing_labels= 9,
            #                                 train=False, partial=True)
            # testloaderOut_cam = get_biased_mnist_dataloader(root = './datasets/MNIST', batch_size=1,
            #                                 data_label_correlation= 1,
            #                                 n_confusing_labels= 9,
            #                                 train=False, partial=True)
        else:
            testsetout = torchvision.datasets.ImageFolder("/nobackup-slow/dataset/background/{}/val".format(out_dataset),
                                        transform= val_transform)
            testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=args.ood_batch_size,
                                             shuffle=True, num_workers=2)
            testloaderOut_cam = torch.utils.data.DataLoader(testsetout, batch_size= 1,
                                             shuffle=True, num_workers=2)
        if CAM:
            return testloaderOut, testloaderOut_cam
        else: 
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

