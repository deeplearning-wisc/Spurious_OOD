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
import logging
import warnings
warnings.filterwarnings("ignore")
# used for logging to TensorBoard
from tensorboard_logger import configure, log_value

class NeuralLinear(object):
    '''
    the neural linear model
    '''
    def __init__(self, args, model, clsfier, repr_dim, output_dim):

        self.args = args
        self.model = model
        self.clsfier = clsfier
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
        self.train_x = torch.empty(0, 3, 256, 256)
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
                batch_size=64, shuffle=True)

        self.model.eval()
        self.clsfier.eval()
        with torch.no_grad():
            for i, (images, labels) in enumerate(data_loader):
                #images = images.cuda()
                # labels = labels.cuda().float()
                outputs = self.model(images.cuda())
                partial_latent_z = self.clsfier.intermediate_forward(outputs)
                if latent_z == None: 
                    latent_z = partial_latent_z 
                else: 
                    latent_z = torch.cat((latent_z , partial_latent_z), dim = 0)
            self.latent_z = latent_z
            assert len(self.latent_z) == len(self.train_x)
        
    def train_blr(self, train_loader_in, train_loader_out, model, clsfier, criterion, num_classes, optimizer, epoch, save_dir, log):
        """Train for one epoch on the training set"""
        batch_time = AverageMeter()

        out_confs = AverageMeter()
        in_confs = AverageMeter()

        in_losses = AverageMeter()
        out_losses = AverageMeter()
        log.debug("######## Start training NN at epoch {} ########".format(epoch) )

        end = time.time()

        epoch_buffer_in = torch.empty(0, 3, 256, 256)
        epoch_buffer_out = torch.empty(0, 3, 256, 256)
        for i, (in_set, out_set) in enumerate(zip(train_loader_in, train_loader_out)):
            in_len = len(in_set[0])
            out_len = len(out_set[0])
            epoch_buffer_in = torch.cat((epoch_buffer_in, in_set[0]), 0)
            epoch_buffer_out = torch.cat((epoch_buffer_out, out_set[0]), 0)
            in_input = in_set[0].cuda() # 64, 3, 244, 244
            in_target = torch.cat( (in_set[1], torch.zeros(in_len, 1)), dim = 1).cuda().float()

            out_input = out_set[0].to(in_input.device) # 128, 3, 244, 244
            out_target = torch.nn.functional.one_hot(out_set[1], num_classes + 1).to(in_target.device)

            model.train()
            clsfier.train()

            cat_input = torch.cat((in_input, out_input), 0) # 192, 3, 244, 244
            cat_output = clsfier(model(cat_input))  # 192, 20
            # cat_target = torch.cat((in_target, out_target), 0)

            ground_truth_logit = torch.full( (cat_output.shape[0], 1), -1 * self.args.conf, dtype = torch.float)
            ground_truth_logit[-len(out_input):] = self.args.conf

            in_output = cat_output[:in_len] #64, 10

            # # train NN with sigmoid 
            in_conf = torch.sigmoid(in_output)[:,-1].mean()
            in_confs.update(in_conf.data, in_len)
            in_loss = criterion(in_output, in_target.to(in_output.device))

            out_output = cat_output[in_len:] # 128, 20
            out_conf = torch.sigmoid(out_output)[:,-1].mean()
            out_confs.update(out_conf.data, out_len)
            # new_out_target = torch.nn.functional.one_hot(out_target, num_classes + 1).float()
            out_loss = criterion(out_output, out_target.to(out_output.device).float()) 

            in_losses.update(in_loss.data, in_len) 
            out_losses.update(out_loss.data, out_len)

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
                    'Out Loss {out_loss.val:.4f} ({out_loss.avg:.4f})\t'
                    'In Conf {in_confs.val:.4f} ({in_confs.avg:.4f})\t'
                    'OOD Conf {out_confs.val:.4f} ({out_confs.avg:.4f})'.format(
                        epoch, i, len(train_loader_in),
                        batch_time=batch_time,
                        in_loss=in_losses, 
                        out_loss=out_losses, out_confs=out_confs,
                        in_confs=in_confs))
        
        self.push_to_cyclic_buffer(epoch_buffer_in, epoch_buffer_out, epoch)
        # log to TensorBoard
        # if self.args.tensorboard:
        #     log_value('nat_train_acc', nat_top1.avg, epoch)

    
    def sample_BDQN(self):
        # Sample sigma^2, and beta conditional on sigma^2
        with torch.no_grad():
            d = self.mu_w[0].shape[0]  #hidden_dim
            for i in range(self.output_dim):
                mus = self.mu_w[i].double() # torch.Size([40])
                covs = self.cov_w[i][np.newaxis, :, :].double() # torch.Size([1, 40, 40])
                try:
                    multivariates = MultivariateNormal(mus, covs[0]).sample().reshape(1, -1)
                    # multivariates = MultivariateNormal(mus, covs[0]).sample().reshape(1, -1)
                except Exception as e:
                    # Sampling could fail if covariance is not positive definite
                    print('Err in Sampling BDQN Details:', e)
                    print('device check: ', mus.device, covs.device)
                    multivariates = MultivariateNormal(torch.zeros(d), torch.eye(d)).sample().reshape(1, -1).to(mus.device)
                if i == 0:
                    beta_s =  multivariates 
                else: 
                    beta_s = torch.cat((beta_s, multivariates), dim = 0)
            self.beta_s = beta_s.float()

    def predict(self, x):
        latent_z = self.model(x)
        latent_z = self.clsfier.intermediate_forward(latent_z)
        return torch.mm(self.beta_s, latent_z.T).T


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