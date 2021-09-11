from __future__ import print_function
import argparse

from torch.utils.data import DataLoader
from datasets.cub_dataset import WaterbirdDataset
from datasets.celebA_dataset import celebAOodDataset, get_celebA_dataloader
import os

import sys
from utils.mahalanobis_lib import get_Mahalanobis_score
from datasets.gaussian_dataset import GaussianDataset
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
# import models.loading_densenet as ld
# import models.initializers as initializers
# import models.bn_initializers as bn_initializers
# import models.densenet as dn
# import models.wideresnet as wn
# import models.gmm as gmmlib
import utils.svhn_loader as svhn
import numpy as np
import time
#import lmdb
from scipy import misc
# from models import CNNModel, res50, res18
# from utils import OODScoreLinfPGDAttack, ConfidenceLinfPGDAttack, MahalanobisLinfPGDAttack, SOFLLinfPGDAttack, metric, sample_estimator, get_Mahalanobis_score, gen_corruction_image
from datasets.color_mnist import get_biased_mnist_dataloader
# from models import CNNModel, res50, res18
from models.resnet_gram import load_model
from models import Resnet

parser = argparse.ArgumentParser(description='OOD Detection Evaluation for various methods')

parser.add_argument('--in-dataset', default="celebA", type=str, help='in-distribution dataset')
parser.add_argument('--exp-name', default = 'erm_new_pretrain_2', type=str, help='help identify checkpoint')
parser.add_argument('--name', '-n', default = 'erm_rebuttal', type=str, help='name of experiment')
parser.add_argument('--method', default='mahalanobis', type=str, help='ood detection method')

parser.add_argument('--epochs', default ="25", type=str,
                    help='number of total epochs to run')
parser.add_argument('--model-arch', default='resnet18_gram', type=str, help='model architecture')
parser.add_argument('--domain-num', default=4, type=int,
                    help='the number of environments for model training')
parser.add_argument('--gpu', default = '1', type = str,
		    help='gpu index')

parser.add_argument('--adv', help='adv ood', action='store_true')
parser.add_argument('--corrupt', help='corrupt', action='store_true')
parser.add_argument('--adv-corrupt', help='adv corrupt', action='store_true')
 
parser.add_argument('--in-dist-only', help='only evaluate in-distribution', action='store_true')
parser.add_argument('--out-dist-only', help='only evaluate out-distribution', action='store_true')

parser.add_argument('--cal-metric', help='calculate metric directly', action='store_true')

parser.add_argument('--epsilon', default=8.0, type=float, help='epsilon')
parser.add_argument('--iters', default=40, type=int,
                    help='attack iterations')
parser.add_argument('--iter-size', default=1.0, type=float, help='attack step size')
parser.add_argument('--severity-level', default=5, type=int, help='severity level')

parser.add_argument('-b', '--batch-size', default=100, type=int,
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
parser.add_argument('--ood-batch-size', default= 100, type=int,
                    help='mini-batch size (default: 400) used for testing')
parser.set_defaults(argument=True)

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


torch.manual_seed(2)
torch.cuda.manual_seed(2)
np.random.seed(2)

def get_msp_score(inputs, model, method_args):
    with torch.no_grad():
        _, outputs = model(inputs)
    scores = np.max(F.softmax(outputs, dim=1).detach().cpu().numpy(), axis=1)

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
    _, outputs = model(Variable(tempInputs))
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

def get_ood_loader(out_dataset, in_dataset = 'color_mnist'):
        # for mnist
        small_transform = transforms.Compose([
                transforms.Resize(32),
                 transforms.CenterCrop(32),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5),
                             std=(0.5, 0.5, 0.5))])
        # for celebA
        scale = 256.0/224.0
        target_resolution = (224, 224)
        large_transform = transforms.Compose([
            transforms.Resize((int(target_resolution[0]*scale), int(target_resolution[1]*scale))),
            transforms.CenterCrop(target_resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        root_dir  = '/nobackup-slow/spurious_ood'

        if in_dataset == 'color_mnist':   
            # if out_dataset == "SVHN":
            #     testsetout = svhn.SVHN(f"{root_dir}/{out_dataset}", split='test',
            #                         transform=small_transform, download=False)
            if out_dataset == 'gaussian':
                testsetout = GaussianDataset(dataset_size =10000, img_size = 32,
                    transform=transforms.Normalize(mean=(0.5, 0.5, 0.5),
                             std=(0.5, 0.5, 0.5)))
            else:
                testsetout = torchvision.datasets.ImageFolder(f"{root_dir}/{out_dataset}",
                                            transform=small_transform)
            subset = torch.utils.data.Subset(testsetout, np.random.choice(len(testsetout), 2000, replace=False))
            testloaderOut = torch.utils.data.DataLoader(subset, batch_size=args.ood_batch_size,
                                             shuffle=True, num_workers=4) 
        elif in_dataset == 'waterbird' or in_dataset == 'celebA':
            if out_dataset == "SVHN":
                testsetout = svhn.SVHN(f"{root_dir}/{out_dataset}", split='test',
                                    transform=large_transform, download=False)
            elif out_dataset == 'gaussian':
                testsetout = GaussianDataset(dataset_size =10000, img_size = 224,
                    transform=transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
            elif  out_dataset == 'celebA_ood':
                testsetout = celebAOodDataset()
                # testloaderOut = get_celebA_ood_dataloader(args)
            else:
                testsetout = torchvision.datasets.ImageFolder(f"{root_dir}/{out_dataset}",
                                            transform=large_transform)
            if out_dataset == 'celebA_ood':
                subset = torch.utils.data.Subset(testsetout, np.random.choice(len(testsetout), 2000, replace=True))
            else:
                subset = torch.utils.data.Subset(testsetout, np.random.choice(len(testsetout), 2000, replace=False))
            testloaderOut = torch.utils.data.DataLoader(subset, batch_size=args.ood_batch_size,
                                                shuffle=True, num_workers=4)           
        else: 
            testsetout = torchvision.datasets.ImageFolder("datasets/ood_datasets/{}".format(out_dataset),
                                        transform= small_transform)
            testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=args.ood_batch_size,
                                             shuffle=True, num_workers=4)
        return testloaderOut

def get_score(inputs, model, method, method_args):
    if method == "msp":
        scores = get_msp_score(inputs, model, method_args)
    elif method == "odin":
        scores = get_odin_score(inputs, model, method_args)
    elif method == "mahalanobis":
        scores = get_mahalanobis_score(inputs, model, method_args)
    return scores

def eval_ood_detector(args, method_args, mode_args):

    in_save_dir = os.path.join(args.base_dir, args.in_dataset, args.method, args.name, args.exp_name, f'{args.epochs}_nat')

    if not os.path.exists(in_save_dir):
        os.makedirs(in_save_dir)

    start = time.time()


    if args.in_dataset == "color_mnist":
        testloaderIn = get_biased_mnist_dataloader(args, root = './datasets/MNIST', batch_size=args.batch_size,
                                            data_label_correlation= args.data_label_correlation,
                                            n_confusing_labels= args.num_classes - 1,
                                            train=False, partial=True, cmap = "1")
                                            
        num_classes = 2

    elif args.in_dataset == "celebA":
        testloaderIn= get_celebA_dataloader(args, split='test')
        num_classes = 2
    
    elif args.in_dataset == "waterbird":
        val_dataset = WaterbirdDataset(data_correlation=args.data_label_correlation, split='test')
        testloaderIn = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
        num_classes = 2

    num_reject_classes = 0
    method_args['num_classes'] = num_classes


    if args.model_arch == "resnet18_gram":
        # model = res18(n_classes=args.num_classes, method=args.name.split("_")[0])
        model = load_model()
    else:
        model = Resnet(n_classes=args.num_classes, model=args.model_arch, method=args.method, domain_num=args.domain_num)
    if args.in_dataset == 'color_mnist':
        cpts_directory = "./checkpoints/{in_dataset}/{name}/{exp}".format(in_dataset=args.in_dataset, name=args.name, exp=args.exp_name)
    elif args.in_dataset == 'waterbird':
        cpts_directory = "./checkpoints/{in_dataset}/{name}/{exp}".format(in_dataset=args.in_dataset, name=args.name, exp=args.exp_name) 
        # cpts_directory = "/nobackup-slow/spurious_ood/checkpoints/{in_dataset}/{name}/{exp}".format(in_dataset=args.in_dataset, name=args.name, exp=args.exp_name)
    else: 
        cpts_directory = "./checkpoints/{in_dataset}/{name}/{exp}".format(in_dataset=args.in_dataset, name=args.name, exp=args.exp_name)   
        # cpts_directory = "/nobackup-slow/spurious_ood/checkpoints/{in_dataset}/{exp}".format(in_dataset=args.in_dataset, exp=args.exp_name)
    
    cpts_dir = os.path.join(cpts_directory, "checkpoint_{epochs}.pth.tar".format(epochs=args.epochs))
    checkpoint = torch.load(cpts_dir)
    state_dict = checkpoint['state_dict_model']
    if torch.cuda.device_count() == 1:
        new_state_dict = {}
        for k, v in state_dict.items():
            k = k.replace("module.", "")
            new_state_dict[k] = v
        state_dict = new_state_dict
    elif torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model, device_ids = [0,1,2,3,4])
    
    model.load_state_dict(state_dict)
    model.eval()
    model.cuda()

    if args.method == "mahalanobis":
        if args.in_dataset == 'color_mnist':
            temp_x = torch.rand(2,3,28,28)
        else:
            temp_x = torch.rand(2,3, 224, 224)
        # temp_x = torch.rand(2,3,32,32)
        temp_x = temp_x.cuda()
        with torch.no_grad():
            if torch.cuda.device_count() > 1:
                temp_list = model.module.feature_list(temp_x)[1]
            else:
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

            scores = get_score(inputs, model, args.method, method_args)

            for score in scores:
                f1.write("{}\n".format(score))


            with torch.no_grad():
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

        
        testloaderOut = get_ood_loader(out_dataset, args.in_dataset)

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
            scores = get_score(inputs, model, args.method, method_args)

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

    if args.in_dataset == 'color_mnist':
        # out_datasets = ['partial_color_mnist_0&1']
        out_datasets = ['partial_color_mnist_0&1', 'dtd', 'iSUN', 'LSUN_resize']
    elif args.in_dataset == 'celebA':
        out_datasets = ['celebA_ood', 'gaussian', 'SVHN', 'iSUN', 'LSUN_resize']
    elif args.in_dataset == 'waterbird':
        out_datasets = [ 'gaussian', 'placesbg', 'SVHN', 'iSUN', 'LSUN_resize', 'dtd']

    print("checking args...")
    print("adv: {}, corrupt: {}, adv_corrput: {})".format(args.adv, args.corrupt, args.adv_corrupt))
    for epoch in args.epochs.split():
        if args.method == 'msp':
                eval_ood_detector(args, method_args,  mode_args)
        elif args.method == "odin":
            method_args['temperature'] = 1000.0
            if args.in_dataset == "CIFAR-10" or args.in_dataset == "waterbird" or args.in_dataset == "celebA":
                method_args['magnitude'] = 0.0014
            elif args.in_dataset == "CIFAR-100":
                method_args['magnitude'] = 0.0028

            eval_ood_detector(args, method_args, mode_args)
        elif args.method == 'mahalanobis':
            sample_mean, precision, lr_weights, lr_bias, magnitude = np.load(os.path.join('output/mahalanobis_hyperparams/', args.in_dataset, args.name, args.exp_name, 'results.npy'), allow_pickle=True)
            regressor = LogisticRegressionCV(cv=2).fit([[0,0,0,0],[0,0,0,0],[1,1,1,1],[1,1,1,1]], [0,0,1,1])
            regressor.coef_ = lr_weights
            regressor.intercept_ = lr_bias

            method_args['sample_mean'] = sample_mean
            method_args['precision'] = precision
            method_args['magnitude'] = magnitude
            method_args['regressor'] = regressor

            eval_ood_detector(args, method_args, mode_args)
