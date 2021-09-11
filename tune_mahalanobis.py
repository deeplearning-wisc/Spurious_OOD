from __future__ import print_function
import argparse
from datasets.cub_dataset import get_waterbird_dataloader
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
# import models.densenet as dn
#import models.wideresnet as wn
import utils.svhn_loader as svhn
import numpy as np
import time
import utils.svhn_loader as svhn
from scipy import misc
from utils import metric, sample_estimator, get_Mahalanobis_score
from datasets.color_mnist import get_biased_mnist_dataloader
from datasets.celebA_dataset import get_celebA_dataloader
# from models import CNNModel, res50, res18
from models.resnet_gram import load_model
from models import Resnet
torch.manual_seed(3)
torch.cuda.manual_seed(3)
np.random.seed(3)

parser = argparse.ArgumentParser(description='Pytorch Detecting Out-of-distribution examples in neural networks')

parser.add_argument('--in-dataset', default="celebA", type=str, help='in-distribution dataset')
parser.add_argument('--exp-name', default = 'erm_new_pretrain_2', type=str, help='help identify checkpoint')
parser.add_argument('--name', '-n', default = 'erm_rebuttal', type=str, help='name of experiment')
parser.add_argument('--method', default='erm', type=str, help='method used for model training')
parser.add_argument('--data_label_correlation', default=0.8, type=float,
                    help='data_label_correlation')
parser.add_argument('--model-arch', default='resnet18_gram', type=str, help='model architecture')

parser.add_argument('--gpu', default = '2', type = str,
		    help='gpu index')
parser.add_argument('--epochs', default=25, type=int,
                    help='number of total epochs to run')

parser.add_argument('-b', '--batch-size', default=50, type=int,
                    help='mini-batch size')
parser.add_argument('--num-classes', default=2, type=int,
                    help='number of classes for model training')
parser.add_argument('--multi-gpu', default=False, type=bool,
                    help='number of cards to use')
# parser.add_argument('--layers', default=100, type=int,
#                     help='total number of layers (default: 100)')

# parser.add_argument('--depth', default=40, type=int,
#                     help='depth of resnet')
# parser.add_argument('--width', default=4, type=int,
#                     help='width of resnet')

parser.set_defaults(argument=True)

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

def print_results(results, stypes):
    mtypes = ['FPR', 'DTERR', 'AUROC', 'AUIN', 'AUOUT']

    for stype in stypes:
        print(' OOD detection method: ' + stype)
        for mtype in mtypes:
            print(' {mtype:6s}'.format(mtype=mtype), end='')
        print('\n{val:6.2f}'.format(val=100.*results[stype]['FPR']), end='')
        print(' {val:6.2f}'.format(val=100.*results[stype]['DTERR']), end='')
        print(' {val:6.2f}'.format(val=100.*results[stype]['AUROC']), end='')
        print(' {val:6.2f}'.format(val=100.*results[stype]['AUIN']), end='')
        print(' {val:6.2f}\n'.format(val=100.*results[stype]['AUOUT']), end='')
        print('')

def tune_mahalanobis_hyperparams():

    print('Tuning hyper-parameters...')
    stypes = ['mahalanobis']

    save_dir = os.path.join('output/mahalanobis_hyperparams/', args.in_dataset, args.name, 'tmp')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if args.in_dataset == "color_mnist":
        trainloaderIn = get_biased_mnist_dataloader(args, root = './datasets/MNIST', batch_size=args.batch_size,
                                            data_label_correlation= args.data_label_correlation,
                                            n_confusing_labels= args.num_classes - 1,
                                            train=True, partial=True, cmap = "1")
        testloaderIn = get_biased_mnist_dataloader(args, root = './datasets/MNIST', batch_size=args.batch_size,
                                            data_label_correlation= args.data_label_correlation,
                                            n_confusing_labels= args.num_classes - 1,
                                            train=False, partial=True, cmap = "1")
        num_classes = 2

    elif args.in_dataset == "celebA":
        trainloaderIn = get_celebA_dataloader(args, split="train")
        testloaderIn = get_celebA_dataloader(args, split='test')
        num_classes = 2
    elif args.in_dataset == "waterbird":
        trainloaderIn = get_waterbird_dataloader(args, data_label_correlation=args.data_label_correlation, split="train")
        testloaderIn = get_waterbird_dataloader(args, data_label_correlation=args.data_label_correlation, split="val")
        num_classes = 2

    if args.model_arch == "resnet18_gram":
        # model = res18(n_classes=args.num_classes, method=args.method)
        model = load_model()
    else:
        model = Resnet(n_classes=args.num_classes, model=args.model_arch, method=args.method, domain_num= 4)

    if args.in_dataset == 'color_mnist':
        cpts_directory = "./checkpoints/{in_dataset}/{name}/{exp}".format(in_dataset=args.in_dataset, name=args.name, exp=args.exp_name)
    elif args.in_dataset == 'waterbird':
        cpts_directory = "/nobackup-slow/spurious_ood/checkpoints/{in_dataset}/{name}/{exp}".format(in_dataset=args.in_dataset, name=args.name, exp=args.exp_name)
    else:    
        # cpts_directory = "/nobackup-slow/spurious_ood/checkpoints/{in_dataset}/{exp}".format(in_dataset=args.in_dataset, exp=args.exp_name)
        cpts_directory = "./checkpoints/{in_dataset}/{name}/{exp}".format(in_dataset=args.in_dataset, name=args.name, exp=args.exp_name)
    cpts_dir = os.path.join(cpts_directory, "checkpoint_{epochs}.pth.tar".format(epochs=args.epochs))
    checkpoint = torch.load(cpts_dir)
    state_dict = checkpoint['state_dict_model']
    if torch.cuda.device_count() == 1:
        new_state_dict = {}
        for k, v in state_dict.items():
            k = k.replace("module.", "")
            new_state_dict[k] = v
        state_dict = new_state_dict
    
    model.load_state_dict(state_dict)

    model.eval()
    model.cuda()

    # set information about feature extaction
    if args.in_dataset == 'color_mnist':
        temp_x = torch.rand(2,3,28,28)
    else:
        temp_x = torch.rand(2,3, 224, 224)
    temp_x = Variable(temp_x).cuda()
    temp_list = model.feature_list(temp_x)[1]
    num_output = len(temp_list)
    feature_list = np.empty(num_output)
    count = 0
    for out in temp_list:
        feature_list[count] = out.size(1)
        count += 1

    print('get sample mean and covariance')
    sample_mean, precision = sample_estimator(model, num_classes, feature_list, trainloaderIn)

    print('train logistic regression model')
    m = 500

    train_in = []
    train_in_label = []
    train_out = []

    val_in = []
    val_in_label = []
    val_out = []

    cnt = 0
    for data, target,_ in testloaderIn:
        data = data.numpy()
        target = target.numpy()
        for x, y in zip(data, target):
            cnt += 1
            if cnt <= m:
                train_in.append(x)
                train_in_label.append(y)
            elif cnt <= 2*m:
                val_in.append(x)
                val_in_label.append(y)

            if cnt == 2*m:
                break
        if cnt == 2*m:
            break

    print('In', len(train_in), len(val_in))

    criterion = nn.CrossEntropyLoss().cuda()
    adv_noise = 0.05

    for i in range(int(m/args.batch_size) + 1):
        if i*args.batch_size >= m:
            break
        data = torch.tensor(train_in[i*args.batch_size:min((i+1)*args.batch_size, m)])
        target = torch.tensor(train_in_label[i*args.batch_size:min((i+1)*args.batch_size, m)])
        data = data.cuda()
        target = target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        _, output = model(data)

        model.zero_grad()
        inputs = Variable(data.data, requires_grad=True).cuda()
        _, output = model(inputs)
        loss = criterion(output, target)
        loss.backward()

        gradient = torch.ge(inputs.grad.data, 0)
        gradient = (gradient.float()-0.5)*2

        adv_data = torch.add(input=inputs.data, other=gradient, alpha=adv_noise)
        adv_data = torch.clamp(adv_data, 0.0, 1.0)

        train_out.extend(adv_data.cpu().numpy())

    for i in range(int(m/args.batch_size) + 1):
        if i*args.batch_size >= m:
            break
        data = torch.tensor(val_in[i*args.batch_size:min((i+1)*args.batch_size, m)])
        target = torch.tensor(val_in_label[i*args.batch_size:min((i+1)*args.batch_size, m)])
        data = data.cuda()
        target = target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        _, output = model(data)

        model.zero_grad()
        inputs = Variable(data.data, requires_grad=True).cuda()
        _, output = model(inputs)
        loss = criterion(output, target)
        loss.backward()

        gradient = torch.ge(inputs.grad.data, 0)
        gradient = (gradient.float()-0.5)*2

        adv_data = torch.add(input=inputs.data, other=gradient, alpha=adv_noise)
        adv_data = torch.clamp(adv_data, 0.0, 1.0)

        val_out.extend(adv_data.cpu().numpy())

    print('Out', len(train_out),len(val_out))

    train_lr_data = []
    train_lr_label = []
    train_lr_data.extend(train_in)
    train_lr_label.extend(np.zeros(m))
    train_lr_data.extend(train_out)
    train_lr_label.extend(np.ones(m))
    train_lr_data = torch.tensor(train_lr_data)
    train_lr_label = torch.tensor(train_lr_label)

    best_fpr = 1.1
    best_magnitude = 0.0

    for magnitude in [0.0, 0.01, 0.005, 0.002, 0.0014, 0.001, 0.0005]:
        train_lr_Mahalanobis = []
        total = 0
        for data_index in range(int(np.floor(train_lr_data.size(0) / args.batch_size))):
            data = train_lr_data[total : total + args.batch_size].cuda()
            total += args.batch_size
            Mahalanobis_scores = get_Mahalanobis_score(data, model, num_classes, sample_mean, precision, num_output, magnitude)
            train_lr_Mahalanobis.extend(Mahalanobis_scores)

        train_lr_Mahalanobis = np.asarray(train_lr_Mahalanobis, dtype=np.float32)
        regressor = LogisticRegressionCV(n_jobs=-1).fit(train_lr_Mahalanobis, train_lr_label)

        print('Logistic Regressor params:', regressor.coef_, regressor.intercept_)

        t0 = time.time()
        f1 = open(os.path.join(save_dir, "confidence_mahalanobis_In.txt"), 'w')
        f2 = open(os.path.join(save_dir, "confidence_mahalanobis_Out.txt"), 'w')

    ########################################In-distribution###########################################
        print("Processing in-distribution images")

        count = 0
        for i in range(int(m/args.batch_size) + 1):
            if i * args.batch_size >= m:
                break
            images = torch.tensor(val_in[i * args.batch_size : min((i+1) * args.batch_size, m)]).cuda()
            # if j<1000: continue
            batch_size = images.shape[0]
            Mahalanobis_scores = get_Mahalanobis_score(images, model, num_classes, sample_mean, precision, num_output, magnitude)
            confidence_scores= regressor.predict_proba(Mahalanobis_scores)[:, 1]

            for k in range(batch_size):
                f1.write("{}\n".format(-confidence_scores[k]))

            count += batch_size
            print("{:4}/{:4} images processed, {:.1f} seconds used.".format(count, m, time.time()-t0))
            t0 = time.time()

    ###################################Out-of-Distributions#####################################
        t0 = time.time()
        print("Processing out-of-distribution images")
        count = 0

        for i in range(int(m/args.batch_size) + 1):
            if i * args.batch_size >= m:
                break
            images = torch.tensor(val_out[i * args.batch_size : min((i+1) * args.batch_size, m)]).cuda()
            # if j<1000: continue
            batch_size = images.shape[0]

            Mahalanobis_scores = get_Mahalanobis_score(images, model, num_classes, sample_mean, precision, num_output, magnitude)

            confidence_scores= regressor.predict_proba(Mahalanobis_scores)[:, 1]

            for k in range(batch_size):
                f2.write("{}\n".format(-confidence_scores[k]))

            count += batch_size
            print("{:4}/{:4} images processed, {:.1f} seconds used.".format(count, m, time.time()-t0))
            t0 = time.time()

        f1.close()
        f2.close()

        results = metric(save_dir, stypes)
        print_results(results, stypes)
        fpr = results['mahalanobis']['FPR']
        if fpr < best_fpr:
            best_fpr = fpr
            best_magnitude = magnitude
            best_regressor = regressor

    print('Best Logistic Regressor params:', best_regressor.coef_, best_regressor.intercept_)
    print('Best magnitude', best_magnitude)

    return sample_mean, precision, best_regressor, best_magnitude

if __name__ == '__main__':
    sample_mean, precision, best_regressor, best_magnitude = tune_mahalanobis_hyperparams()
    print('saving results...')
    save_dir = os.path.join('output/mahalanobis_hyperparams/', args.in_dataset, args.name, args.exp_name)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    np.save(os.path.join(save_dir, 'results'), np.array([sample_mean, precision, best_regressor.coef_, best_regressor.intercept_, best_magnitude]))