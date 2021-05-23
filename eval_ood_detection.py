from __future__ import print_function
import argparse
import os

import sys

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from sklearn.linear_model import LogisticRegressionCV
import models.resnet as rn
import models.loading_densenet as ld
import models.initializers as initializers
import models.bn_initializers as bn_initializers
import models.densenet as dn
import models.wideresnet as wn
import models.gmm as gmmlib
import utils.svhn_loader as svhn
import numpy as np
import time
#import lmdb
from scipy import misc
from models import CNNModel, res50, res18
# from utils import OODScoreLinfPGDAttack, ConfidenceLinfPGDAttack, MahalanobisLinfPGDAttack, SOFLLinfPGDAttack, metric, sample_estimator, get_Mahalanobis_score, gen_corruction_image
from datasets.color_mnist import get_biased_mnist_dataloader
from models import CNNModel, res50, res18

parser = argparse.ArgumentParser(description='OOD Detection Evaluation for various methods')

parser.add_argument('--in-dataset', default="color_mnist", type=str, help='in-distribution dataset')
parser.add_argument('--name', default = "erm_r_0_8", type=str,
                    help='neural network name and training set')
parser.add_argument('--epochs', default ="10", type=str,
                    help='number of total epochs to run')
parser.add_argument('--model-arch', default='resnet18', type=str, help='model architecture')

parser.add_argument('--gpu', default = '0', type = str,
		    help='gpu index')

parser.add_argument('--adv', help='adv ood', action='store_true')
parser.add_argument('--corrupt', help='corrupt', action='store_true')
parser.add_argument('--adv-corrupt', help='adv corrupt', action='store_true')
 
parser.add_argument('--in-dist-only', help='only evaluate in-distribution', action='store_true')
parser.add_argument('--out-dist-only', help='only evaluate out-distribution', action='store_true')

parser.add_argument('--method', default='msp', type=str, help='ood detection method')
parser.add_argument('--cal-metric', help='calculate metric directly', action='store_true')

parser.add_argument('--epsilon', default=8.0, type=float, help='epsilon')
parser.add_argument('--iters', default=40, type=int,
                    help='attack iterations')
parser.add_argument('--iter-size', default=1.0, type=float, help='attack step size')
parser.add_argument('--severity-level', default=5, type=int, help='severity level')

parser.add_argument('-b', '--batch-size', default=400, type=int,
                    help='mini-batch size')

parser.add_argument('--base-dir', default='output/ood_scores', type=str, help='result directory')

# parser.add_argument('--layers', default=100, type=int,
#                     help='total number of layers (default: 100)')
# parser.add_argument('--depth', default=40, type=int,
#                     help='depth of resnet')
# parser.add_argument('--width', default=4, type=int,
#                     help='width of resnet')

parser.add_argument('--data_label_correlation', default= 0.8, type=float,
                    help='data_label_correlation')
parser.add_argument('--num-classes', default=2, type=int,
                    help='number of classes for model training')
parser.add_argument('--ood-batch-size', default= 64, type=int,
                    help='mini-batch size (default: 400) used for testing')
parser.add_argument('--multi_gpu', default=False, type=bool,
                    help='if distributed')
parser.set_defaults(argument=True)

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)

def get_msp_score(inputs, model, method_args):
    with torch.no_grad():
        _, outputs = model(inputs)
    scores = np.max(F.softmax(outputs, dim=1).detach().cpu().numpy(), axis=1)

    return scores

def get_sofl_score(inputs, model, method_args):
    num_classes = method_args['num_classes']
    with torch.no_grad():
        outputs = model(inputs)
    scores = -F.softmax(outputs, dim=1)[:, num_classes:].sum(dim=1).detach().cpu().numpy()

    return scores

def get_rowl_score(inputs, model, method_args):
    num_classes = method_args['num_classes']
    with torch.no_grad():
        outputs = model(inputs)
    #scores = -F.softmax(outputs, dim=1)[:, num_classes]
    scores = -1.0 * (outputs.argmax(dim=1)==num_classes).float().detach().cpu().numpy()

    return scores

def get_atom_score(inputs, model, method_args):
    num_classes = method_args['num_classes']
    with torch.no_grad():
        outputs = model(inputs)
    #scores = -F.softmax(outputs, dim=1)[:, num_classes]
    scores = -1.0 * (F.softmax(outputs, dim=1)[:,-1]).float().detach().cpu().numpy()

    return scores

def get_odin_score(inputs, model, method_args):
    # Calculating the perturbation we need to add, that is,
    # the sign of gradient of cross entropy loss w.r.t. input

    temper = method_args['temperature']
    noiseMagnitude1 = method_args['magnitude']

    criterion = nn.CrossEntropyLoss()
    inputs = Variable(inputs, requires_grad = True)
    _, outputs = model(inputs)

    maxIndexTemp = np.argmax(outputs.data.cpu().numpy(), axis=1)

    # Using temperature scaling
    outputs = outputs / temper

    labels = Variable(torch.LongTensor(maxIndexTemp).cuda())
    loss = criterion(outputs, labels)
    loss.backward()

    # Normalizing the gradient to binary in {0, 1}
    gradient =  torch.ge(inputs.grad.data, 0) # returns a tensor of boolean var
    gradient = (gradient.float() - 0.5) * 2

    # Adding small perturbations to images
    tempInputs = torch.add(inputs.data,  -noiseMagnitude1, gradient)
    outputs = model(Variable(tempInputs))
    outputs = outputs / temper
    # Calculating the confidence after adding perturbations
    nnOutputs = outputs.data.cpu()
    nnOutputs = nnOutputs.numpy()
    nnOutputs = nnOutputs - np.max(nnOutputs, axis=1, keepdims=True)
    nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs), axis=1, keepdims=True)
    scores = np.max(nnOutputs, axis=1)

    return scores

def get_mahalanobis_score(inputs, model, method_args):
    num_classes = method_args['num_classes']
    sample_mean = method_args['sample_mean']
    precision = method_args['precision']
    magnitude = method_args['magnitude']
    regressor = method_args['regressor']
    num_output = method_args['num_output']

    Mahalanobis_scores = get_Mahalanobis_score(inputs, model, num_classes, sample_mean, precision, num_output, magnitude)
    scores = -regressor.predict_proba(Mahalanobis_scores)[:, 1]

    return scores

def get_score(inputs, model, method, method_args):
    if method == "msp":
        scores = get_msp_score(inputs, model, method_args)
    elif method == "odin":
        scores = get_odin_score(inputs, model, method_args)
    elif method == "mahalanobis":
        scores = get_mahalanobis_score(inputs, model, method_args)
    elif method == "sofl":
        scores = get_sofl_score(inputs, model, method_args)
    elif method == "rowl":
        scores = get_rowl_score(inputs, model, method_args)
    elif method == "atom":
        scores = get_atom_score(inputs, model, method_args)

    return scores

def eval_ood_detector(base_dir, in_dataset, out_datasets, batch_size, method, method_args, name, epochs, adv, corrupt, adv_corrupt, adv_args, mode_args):

    in_save_dir = os.path.join(base_dir, in_dataset, method, name, f'{epochs}_nat')

    if not os.path.exists(in_save_dir):
        os.makedirs(in_save_dir)

    start = time.time()

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    if args.in_dataset == "CIFAR-10":
        normalizer = transforms.Normalize((125.3/255, 123.0/255, 113.9/255), (63.0/255, 62.1/255.0, 66.7/255.0))
        testset = torchvision.datasets.CIFAR10(root='./datasets/cifar10', train=False, download=True, transform=transform)
        testloaderIn = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=True, num_workers=4)
        num_classes = 10
        num_reject_classes = 5
    elif args.in_dataset == "CIFAR-100":
        normalizer = transforms.Normalize((125.3/255, 123.0/255, 113.9/255), (63.0/255, 62.1/255.0, 66.7/255.0))
        testset = torchvision.datasets.CIFAR100(root='./datasets/cifar100', train=False, download=True, transform=transform)
        testloaderIn = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=True, num_workers=4)
        num_classes = 100
        num_reject_classes = 10

    elif in_dataset == "SVHN":
        normalizer = None
        testset = svhn.SVHN('datasets/svhn/', split='test',
                              transform=transforms.ToTensor(), download=False)
        testloaderIn = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=True, num_workers=4)
        num_classes = 10
        num_reject_classes = 5
    elif args.in_dataset == "IN-9":
        normalize = transforms.Normalize((125.3/255, 123.0/255, 113.9/255), (63.0/255, 62.1/255.0, 66.7/255.0))
        testset = torchvision.datasets.ImageFolder(root="/nobackup-slow/dataset/background/original/val", 
                  transform=transforms.Compose([transforms.Scale(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        normalize]))
        testloaderIn = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=True, num_workers=4)
        num_classes = 9
        num_reject_classes = 5

    elif args.in_dataset == "color_mnist":
        # val_loader  = get_ood_loader("0_background")
        testloaderIn = get_biased_mnist_dataloader(args, root = './datasets/MNIST', batch_size=args.batch_size,
                                            data_label_correlation= args.data_label_correlation,
                                            n_confusing_labels= args.num_classes - 1,
                                            train=False, partial=True, cmap = "1")
        num_classes = args.num_classes
        num_reject_classes = 1

    if method != "sofl":
        num_reject_classes = 0

    if method == "rowl" or method == "atom":
        num_reject_classes = 1

    method_args['num_classes'] = num_classes

    # if args.model_arch == "resnet18":
    #     model = res18(n_classes=args.num_classes, method=args.method)
    if args.model_arch == "general_model":
        model = CNNModel(num_classes=args.num_classes, bn_init=True, method=args.method)
    elif args.model_arch == 'densenet':
        model = dn.DenseNet3(args.layers, num_classes + num_reject_classes, normalizer=normalizer)
    elif args.model_arch == 'wideresnet':
        model = wn.WideResNet(args.depth, num_classes + num_reject_classes, widen_factor=args.width, normalizer=normalizer)
    elif args.model_arch == 'densenet_ccu':
        model = dn.DenseNet3(args.layers, num_classes + num_reject_classes, normalizer=normalizer)
        gmm = torch.load("checkpoints/{in_dataset}/{name}/".format(in_dataset=args.in_dataset, name=args.name) + 'in_gmm.pth.tar')
        gmm.alpha = nn.Parameter(gmm.alpha)
        gmm_out = torch.load("checkpoints/{in_dataset}/{name}/".format(in_dataset=args.in_dataset, name=args.name) + 'out_gmm.pth.tar')
        gmm_out.alpha = nn.Parameter(gmm.alpha)
        whole_model = gmmlib.DoublyRobustModel(model, gmm, gmm_out, loglam = 0., dim=3072, classes=num_classes)
    elif args.model_arch == "resnet18":
        model = res18(n_classes=args.num_classes, method=args.name.split("_")[0])
    else:
        assert False, 'Not supported model arch: {}'.format(args.model_arch)

    # if args.in_dataset == "color_mnist":
    #     checkpoint = torch.load("./checkpoints/{in_dataset}_res18/{name}/checkpoint_{epochs}.pth.tar".format(in_dataset=in_dataset, name=name, epochs=epochs))
    # else:
    checkpoint = torch.load("./checkpoints/{in_dataset}/{name}/checkpoint_{epochs}.pth.tar".format(in_dataset=in_dataset, name=name, epochs=epochs))
    
    if args.model_arch == 'densenet_ccu':
        whole_model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint['state_dict_model'])
    
    model.eval()
    model.cuda()

    if method == "mahalanobis":
        temp_x = torch.rand(2,3,32,32)
        temp_x = Variable(temp_x).cuda()
        temp_list = model.feature_list(temp_x)[1]
        num_output = len(temp_list)
        method_args['num_output'] = num_output

    if not mode_args['out_dist_only']:
        t0 = time.time()

        f1 = open(os.path.join(in_save_dir, "in_scores.txt"), 'w')
        g1 = open(os.path.join(in_save_dir, "in_labels.txt"), 'w')

    ########################################In-distribution###########################################
        print("Processing in-distribution images")

        N = len(testloaderIn.dataset)
        count = 0
        for j, data in enumerate(testloaderIn):
            images, labels,_ = data
            images = images.cuda()
            labels = labels.cuda()
            curr_batch_size = images.shape[0]

            inputs = images

            scores = get_score(inputs, model, method, method_args)

            for score in scores:
                f1.write("{}\n".format(score))

            if method == "rowl":
                outputs = F.softmax(model(inputs), dim=1)
                outputs = outputs.detach().cpu().numpy()
                preds = np.argmax(outputs, axis=1)
                confs = np.max(outputs, axis=1)
            else:
                _, outputs= model(inputs)
                outputs = F.softmax(outputs[:, :num_classes], dim=1)
                outputs = outputs.detach().cpu().numpy()
                preds = np.argmax(outputs, axis=1)
                confs = np.max(outputs, axis=1)

            for k in range(preds.shape[0]):
                g1.write("{} {} {}\n".format(labels[k], preds[k], confs[k]))

            count += curr_batch_size
            print("{:4}/{:4} images processed, {:.1f} seconds used.".format(count, N, time.time()-t0))
            t0 = time.time()

        f1.close()
        g1.close()

    if mode_args['in_dist_only']:
        return

    for out_dataset in out_datasets:

        out_save_dir = os.path.join(in_save_dir, out_dataset)

        if not os.path.exists(out_save_dir):
            os.makedirs(out_save_dir)

        f2 = open(os.path.join(out_save_dir, "out_scores.txt"), 'w')

        if not os.path.exists(out_save_dir):
            os.makedirs(out_save_dir)

        val_transform = transforms.Compose([
                transforms.Resize(28),
                 transforms.CenterCrop(28),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5),
                             std=(0.5, 0.5, 0.5))])
        if out_dataset == 'SVHN':
            testsetout = svhn.SVHN('datasets/ood_datasets/svhn/', split='test',
                                  transform=val_transform, download=False)
            testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=batch_size,
                                             shuffle=True, num_workers=2)
        elif out_dataset == 'dtd':
            testsetout = torchvision.datasets.ImageFolder(root="datasets/ood_datasets/dtd/images",
                                        transform=val_transform)
            testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=batch_size, shuffle=True,
                                                     num_workers=2)
        elif out_dataset == 'places365':
            testsetout = torchvision.datasets.ImageFolder(root="/nobackup-slow/dataset/places365",
                transform=transforms.Compose([transforms.Resize(32), transforms.CenterCrop(32), transforms.ToTensor()]))
            subset = torch.utils.data.Subset(testsetout, np.random.choice(len(testsetout), 10000, replace=False))
            testloaderOut = torch.utils.data.DataLoader(subset, batch_size=batch_size,
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
        elif 'mnist_5_9' in out_dataset:
            val_transform = transforms.Compose([
                transforms.Grayscale(3),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5),
                             std=(0.5, 0.5, 0.5)),
                ])
            dataset = torchvision.datasets.MNIST(root="./datasets", train=False, transform=val_transform)
            idx = dataset.targets >= 5
            dataset.targets = dataset.targets[idx]
            dataset.data = dataset.data[idx]
            testloaderOut = torch.utils.data.DataLoader(dataset, batch_size=args.ood_batch_size,
                                             shuffle=True, num_workers=2)            
        else:
            testsetout = torchvision.datasets.ImageFolder("./datasets/ood_datasets/{}".format(out_dataset),
                                        transform=val_transform)
            testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=batch_size,
                                             shuffle=True, num_workers=2)

    ###################################Out-of-Distributions#####################################
        t0 = time.time()
        print("Processing out-of-distribution images")

        N = len(testloaderOut.dataset)
        count = 0
        for j, data in enumerate(testloaderOut):

            images, labels = data
            inputs = images.cuda()
            labels = labels.cuda()
            curr_batch_size = images.shape[0]
            scores = get_score(inputs, model, method, method_args)

            for score in scores:
                f2.write("{}\n".format(score))

            count += curr_batch_size
            print("{:4}/{:4} images processed, {:.1f} seconds used.".format(count, N, time.time()-t0))
            t0 = time.time()

        f2.close()

    return

if __name__ == '__main__':
    method_args = dict()
    adv_args = dict()
    mode_args = dict()

    adv_args['epsilon'] = args.epsilon
    adv_args['iters'] = args.iters
    adv_args['iter_size'] = args.iter_size
    adv_args['severity_level'] = args.severity_level

    mode_args['in_dist_only'] = args.in_dist_only
    mode_args['out_dist_only'] = args.out_dist_only

    if args.in_dataset == "CIFAR-10" or args.in_dataset == "CIFAR-100":
        out_datasets = ['places365','LSUN', 'LSUN_resize', 'iSUN', 'dtd', 'SVHN']
        # out_datasets = ['ocean', 'fox']
    elif args.in_dataset == "SVHN":
        out_datasets = ['LSUN', 'LSUN_resize', 'iSUN', 'dtd']
    elif args.in_dataset == 'color_mnist':
        out_datasets = ['dtd', 'iSUN', 'LSUN_resize']
        # out_datasets = ['dtd','SVHN', 'iSUN','LSUN_resize']
        # out_datasets = ['mnist_5_9']
        # out_datasets = ['SVHN']
        # out_datasets = ['partial_color_mnist_0&1','SVHN']

    print("checking args...")
    print("adv: {}, corrupt: {}, adv_corrput: {})".format(args.adv, args.corrupt, args.adv_corrupt))
    for epoch in args.epochs.split():
        if args.method == 'msp':
                eval_ood_detector(args.base_dir, args.in_dataset, out_datasets, args.batch_size, args.method, method_args, args.name, epoch, args.adv, args.corrupt, args.adv_corrupt, adv_args, mode_args)
        elif args.method == "odin":
            method_args['temperature'] = 1000.0
            if args.in_dataset == "CIFAR-10":
                method_args['magnitude'] = 0.0014
            elif args.in_dataset == "CIFAR-100":
                method_args['magnitude'] = 0.0028

            eval_ood_detector(args.base_dir, args.in_dataset, out_datasets, args.batch_size, args.method, method_args, args.name, epoch, args.adv, args.corrupt, args.adv_corrupt, adv_args, mode_args)
        elif args.method == 'mahalanobis':
            sample_mean, precision, lr_weights, lr_bias, magnitude = np.load(os.path.join('output/mahalanobis_hyperparams/', args.in_dataset, args.name, 'results.npy'), allow_pickle=True)
            regressor = LogisticRegressionCV(cv=2).fit([[0,0,0,0],[0,0,0,0],[1,1,1,1],[1,1,1,1]], [0,0,1,1])
            regressor.coef_ = lr_weights
            regressor.intercept_ = lr_bias

            method_args['sample_mean'] = sample_mean
            method_args['precision'] = precision
            method_args['magnitude'] = magnitude
            method_args['regressor'] = regressor

            eval_ood_detector(args.base_dir, args.in_dataset, out_datasets, args.batch_size, args.method, method_args, args.name, epoch, args.adv, args.corrupt, args.adv_corrupt, adv_args, mode_args)
        elif args.method == 'sofl':
            eval_ood_detector(args.base_dir, args.in_dataset, out_datasets, args.batch_size, args.method, method_args, args.name, epoch, args.adv, args.corrupt, args.adv_corrupt, adv_args, mode_args)
        elif args.method == 'rowl':
            eval_ood_detector(args.base_dir, args.in_dataset, out_datasets, args.batch_size, args.method, method_args, args.name, epoch, args.adv, args.corrupt, args.adv_corrupt, adv_args, mode_args)
        elif args.method == 'atom':
                eval_ood_detector(args.base_dir, args.in_dataset, out_datasets, args.batch_size, args.method, method_args, args.name, epoch, args.adv, args.corrupt, args.adv_corrupt, adv_args, mode_args)
