import argparse
import os

import sys

import shutil
import time
import pickle
import json


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
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.lowrank_multivariate_normal import LowRankMultivariateNormal

from scipy.stats import invgamma

import models.densenet as dn
import models.wideresnet as wn
from utils import LinfPGDAttack, TinyImages

import warnings
warnings.filterwarnings("ignore")
# used for logging to TensorBoard
from tensorboard_logger import configure, log_value
import logging

class NeuralLinear(object):
    '''
    the neural linear model
    '''
    def __init__(self, args, model, repr_dim, output_dim):

        self.args = args
        self.model = model
        self.output_dim = output_dim
        self.repr_dim = repr_dim
        self.a = torch.tensor([args.a0 for _ in range(self.output_dim)]).cuda()
        self.b = torch.tensor([args.b0 for _ in range(self.output_dim)]).cuda()

        # self.lambda_prior = args.lambda_prior
        # self.mu = [np.zeros(self.repr_dim) for _ in range(self.output_dim)]
        # self.cov = [(1.0 / self.lambda_prior) * torch.eye(self.repr_dim)
        #          for _ in range(self.output_dim)]

        #Update formula in BDQN
        self.sigma = args.sigma # W prior variance
        self.sigma_n = args.sigma_n # noise variacne
        self.eye = torch.eye(self.repr_dim).cuda()
        self.mu_w = torch.normal(0, 0.01, size = (self.output_dim, self.repr_dim)).cuda()
        #self.cov_w = np.random.normal(loc = 0, scale = 1, size = (self.output_dim, self.repr_dim, self.repr_dim)) + args.var_scale * self.eye 
        cov_w = np.array([self.sigma*np.eye(self.repr_dim) for _ in range(self.output_dim)])
        self.cov_w = torch.from_numpy(cov_w).cuda()

        self.beta_s = None
        self.latent_z = None
        self.train_x = torch.empty(0, 3, 32, 32)
        self.train_y = torch.empty(0, 1)

    def push_to_cyclic_buffer(self, epoch_buf_in, epoch_buf_out, epoch):
        BUF_SIZE = 4
        in_len = epoch_buf_in.shape[0]
        out_len = epoch_buf_out.shape[0]
        ground_truth_in = torch.full( (in_len, 1), -1 * self.args.conf, dtype = torch.float)
        ground_truth_out = torch.full( (out_len, 1), self.args.conf, dtype = torch.float)
        if epoch < 4:
            self.train_x = torch.cat((self.train_x, epoch_buf_in), dim = 0)
            self.train_x = torch.cat((self.train_x, epoch_buf_out), dim = 0)
            self.train_y = torch.cat((self.train_y, ground_truth_in), dim = 0)
            self.train_y = torch.cat((self.train_y, ground_truth_out), dim = 0)
        else:
            cyclic_id = epoch % BUF_SIZE
            replace_idx = np.arange(cyclic_id * (out_len + in_len) + in_len, (cyclic_id + 1) * (out_len + in_len))
            self.train_x[replace_idx] =  epoch_buf_out
            self.train_y[replace_idx] =  ground_truth_out

    def update_representation(self):
        latent_z = None
        print('begin updating representation')
        data_loader = torch.utils.data.DataLoader(
                SimpleDataset(self.train_x, self.train_y),
                batch_size=64, shuffle = False)

        self.model.eval()
        with torch.no_grad():
            for i, (images, labels) in enumerate(data_loader):
                #images = images.cuda()
                # labels = labels.cuda().float()
                partial_latent_z = self.model.get_representation(images.cuda())
                # outputs = self.model(images.cuda())
                # partial_latent_z = self.clsfier.module.intermediate_forward(outputs)
                if latent_z == None: 
                    latent_z = partial_latent_z 
                else: 
                    latent_z = torch.cat((latent_z , partial_latent_z), dim = 0)
            self.latent_z = latent_z
            assert len(self.latent_z) == len(self.train_x)
    
    def train_blr(self, train_loader_in, train_loader_out, model, criterion, num_classes, optimizer, epoch, save_dir, log):
        """Train for one epoch on the training set"""
        batch_time = AverageMeter()

        out_confs = AverageMeter()
        in_confs = AverageMeter()

        in_losses = AverageMeter()
        out_losses = AverageMeter()
        nat_top1 = AverageMeter()
        log.debug("######## Start training NN at epoch {} ########".format(epoch) )
        end = time.time()

        epoch_buffer_in = torch.empty(0, 3, 32, 32)
        epoch_buffer_out = torch.empty(0, 3, 32, 32)
        for i, (in_set, out_set) in enumerate(zip(train_loader_in, train_loader_out)):
            in_len = len(in_set[0])
            out_len = len(out_set[0])
            epoch_buffer_in = torch.cat((epoch_buffer_in, in_set[0]), 0)
            epoch_buffer_out = torch.cat((epoch_buffer_out, out_set[0]), 0)
            in_input = in_set[0].cuda() # 64, 3, 32, 32
            in_target = in_set[1].cuda() # 64
            out_input = out_set[0].cuda() # 128, 3, 32, 32
            out_target = out_set[1].cuda()  # 128

            #ground_truth_logit = torch.full( (cat_input.shape[0], 1), -1 * self.args.conf, dtype = torch.float)
            #ground_truth_logit[-len(out_input):] = self.args.conf

            model.train()

            # assert in_input.device == out_input.device and in_target.device == out_target.device

            # cat_input = torch.cat((in_input, out_input), 0) # 192, 3, 32, 32
            # cat_output = model(cat_input)   # 192, 10
            # in_output = cat_output[:in_len] 

            # train NN with softmax
            # in_conf = F.softmax(in_output, dim=1)[:,-1].mean()
            # in_confs.update(in_conf.data, in_len)
            # in_loss = criterion(in_output, in_target.to(in_output.device))

            # out_output = cat_output[in_len:] # 128, 10
            # out_conf = F.softmax(out_output, dim=1)[:,-1].mean()
            # out_confs.update(out_conf.data, out_len)
            # out_loss = criterion(out_output, out_target.to(out_output.device)) 


            # Try 
            in_output = model( in_input )
            out_output = model(out_input )
            
            in_conf = F.softmax(in_output, dim=1)[:,-1].mean()
            in_confs.update(in_conf.data, in_len)
            in_loss = criterion(in_output, in_target)

            out_conf = F.softmax(out_output, dim=1)[:,-1].mean()
            out_confs.update(out_conf.data, out_len)
            out_loss = criterion(out_output, out_target)
           #

            in_losses.update(in_loss.data, in_len) 
            out_losses.update(out_loss.data, out_len)

            nat_prec1 = accuracy(in_output[:,:num_classes].data, in_target, topk=(1,))[0]
            nat_top1.update(nat_prec1, in_len)

            loss = in_loss + self.args.beta * out_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % self.args.print_freq == 0:
                log.debug('Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'In Loss {in_loss.val:.4f} ({in_loss.avg:.4f})\t'
                    'Prec@1 {nat_top1.val:.3f} ({nat_top1.avg:.3f})\t'
                    'Out Loss {out_loss.val:.4f} ({out_loss.avg:.4f})\t'
                    'In Conf {in_confs.val:.4f} ({in_confs.avg:.4f})\t'
                    'OOD Conf {out_confs.val:.4f} ({out_confs.avg:.4f})'.format(
                        epoch, i, len(train_loader_in),
                        batch_time=batch_time,
                        in_loss=in_losses, nat_top1=nat_top1,
                        out_loss=out_losses, out_confs=out_confs,
                        in_confs=in_confs))


        #self.push_to_cyclic_buffer(epoch_buffer_in, epoch_buffer_out, epoch)
    
    def sample_BDQN(self):
        # Sample sigma^2, and beta conditional on sigma^2
        with torch.no_grad():
            d = self.mu_w[0].shape[0]  #hidden_dim
            try:
                for i in range(self.output_dim):
                    mus = self.mu_w[i].double() # torch.Size([40])
                    covs = self.cov_w[i][np.newaxis, :, :].double() # torch.Size([1, 40, 40])
                    multivariates = MultivariateNormal(mus, covs[0]).sample().reshape(1, -1)
                    if i == 0:
                        beta_s =  multivariates 
                    else: 
                        beta_s = torch.cat((beta_s, multivariates), dim = 0)
            except Exception as e:
                # Sampling could fail if covariance is not positive definite
                # Todo: Fix This
                # print("err: ", e)
                print('Err in Sampling BDQN Details:', e)
                multivariates = MultivariateNormal(torch.zeros(d), torch.eye(d)).sample().reshape(1, -1)
                if i == 0:
                    beta_s =  multivariates 
                else: 
                    beta_s = torch.cat((beta_s, multivariates), dim = 0)
            self.beta_s = beta_s.float()
            # print("beta: ", self.beta_s)

    def predict(self, x):
        latent_z = self.model.get_representation(x)
        return torch.matmul(self.beta_s, latent_z.T).T 


    def update_bays_reg_BDQN(self, log):
        with torch.no_grad():    
            log.debug("######## Start updating bayesian linear layer ########")
            # Update action posterior with formulas: \beta | z,y ~ N(mu_q, cov_q)
            z = self.latent_z.double().cuda()
            y = self.train_y.squeeze().cuda() 
            s = torch.matmul(z.T, z)

            #inv = np.linalg.inv((s/self.sigma_n + 1/self.sigma*self.eye))
            A = s/self.sigma_n + 1/self.sigma*self.eye
            B = torch.matmul(z.T, y.double())/self.sigma_n
            #A = torch.as_tensor(s/self.sigma_n + 1/self.sigma*self.eye)
            #B = torch.as_tensor(np.dot(z.T, y))/self.sigma_n
            A_eig_val, A_eig_vec = torch.symeig(A, eigenvectors=True)
            log.debug("Before inverse. eigenvalue (pd): {}".format(A_eig_val[:20]))
            log.debug("Before inverse. eigenvalue (pd): {}".format(A_eig_val[-20:]))
            A = A.detach().cpu().numpy()
            # inv = torch.inverse(A)
            inv = np.linalg.inv(A)
            inv = torch.from_numpy(inv).cuda()
            self.mu_w[0] = torch.matmul(inv, B).squeeze()
            temp_cov = self.sigma*inv        
            eig_val, eig_vec = torch.symeig(temp_cov, eigenvectors=True)
            log.debug("After inverse. eigenvalue (pd): {}".format(eig_val[:20]) )
            log.debug("After inverse. eigenvalue (pd): {}".format(eig_val[-20:]) )
            if torch.any(eig_val < 0):
                self.cov_w[0] = torch.matmul(torch.matmul(eig_vec, torch.diag(torch.abs(eig_val))), torch.t(eig_vec))
            else:
                self.cov_w[0] = temp_cov
            print('singularity check: matrix not singular.')

  

    def validate(self, val_loader, model, criterion, epoch, log):
        """Perform validation on the validation set"""
        log.debug("######## Start validating at epoch {} ########".format(epoch))
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()

        # switch to evaluate mode
        model.eval()

        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            input = input.cuda()
            target = target.cuda()
            # compute output
            output = model(input)
            # validate NN with softmax
            loss = criterion(output, target.to(output.device))
            # validate NN with sigmoid
            # new_target = torch.nn.functional.one_hot(target, model.module.output_dim).float()
            # loss = criterion(output, new_target.to(output.device))
            # measure accuracy and record loss
            prec1 = accuracy(output.data, target, topk=(1,))[0]
            losses.update(loss.data, input.size(0))
            top1.update(prec1, input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % self.args.print_freq == 0:
                log.debug('Test: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                        i, len(val_loader), batch_time=batch_time, loss=losses,
                        top1=top1))

        log.debug(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
        # log to TensorBoard
        if self.args.tensorboard:
            log_value('val_loss', losses.avg, epoch)
            log_value('val_acc', top1.avg, epoch)
        return top1.avg


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

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t().to(target.device)
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class SimpleDataset(torch.utils.data.Dataset):
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