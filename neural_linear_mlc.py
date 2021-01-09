import argparse
import os

import sys

import shutil
import time
import pickle
import json

from sklearn import metrics
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
    def __init__(self, args, model, clsfier, repr_dim, output_dim, num_classes):

        self.args = args
        self.model = model
        self.clsfier = clsfier
        self.output_dim = output_dim
        self.repr_dim = repr_dim
        self.num_classes = num_classes

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
        self.train_x = torch.empty(0, 3, 224, 224)
        self.train_y = torch.empty(0, 1)

    def push_to_cyclic_buffer(self, epoch_buf_in, epoch_buf_out, epoch):
        BUF_SIZE = 4
        in_len = epoch_buf_in.shape[0]
        out_len = epoch_buf_out.shape[0]
        ground_truth_in = torch.full( (in_len, 1), -2 * self.args.conf, dtype = torch.float)
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
        # print("clc")

    def update_representation(self):
        latent_z = None
        print('begin updating representation')
        data_loader = torch.utils.data.DataLoader(
                SimpleDataset(self.train_x, self.train_y),
                batch_size=64, shuffle=False)

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
        
    def train_blr(self, train_loader_in, train_loader_out, criterion, optimizer, epoch, save_dir, log):
        """Train for one epoch on the training set"""
        batch_time = AverageMeter()

        out_confs = AverageMeter()
        in_confs = AverageMeter()

        in_losses = AverageMeter()
        in_out_losses = AverageMeter()
        out_losses = AverageMeter()
        log.debug("######## Start training NN at epoch {} ########".format(epoch) )

        end = time.time()

        epoch_buffer_in = torch.empty(0, 3, 224, 224)
        epoch_buffer_out = torch.empty(0, 3, 224, 224)

        for i, (in_set, out_set) in enumerate(zip(train_loader_in, train_loader_out)):
            in_len = len(in_set[0])
            out_len = len(out_set[0])
            epoch_buffer_in = torch.cat((epoch_buffer_in, in_set[0]), 0)
            epoch_buffer_out = torch.cat((epoch_buffer_out, out_set[0]), 0)

            in_input = in_set[0].cuda() # 64, 3, 244, 244
            in_target = torch.cat( (in_set[1], torch.zeros(in_len, 1)), dim = 1).cuda().float()
            out_input = out_set[0].to(in_input.device) # 128, 3, 244, 244
            out_target = torch.nn.functional.one_hot(out_set[1], self.num_classes + 1).to(in_target.device)

            # cat_input = torch.cat((in_input, out_input), 0) # 192, 3, 244, 244
            #modified cat_input
            NUM_DEVICE = 4
            id_idx = np.arange(in_len//NUM_DEVICE)
            ood_mask = np.ones(in_len + out_len, dtype=bool)
            for j in range(1, NUM_DEVICE ):
                id_idx = np.hstack([id_idx, np.arange((in_len + out_len)//NUM_DEVICE * j, 
                                    (in_len + out_len)//NUM_DEVICE  * j + in_len//NUM_DEVICE )])
            ood_mask[id_idx] = False 
            ood_idx = np.arange(in_len + out_len)[ood_mask]
            cat_input = torch.empty(in_len + out_len, 3, 224, 224).cuda()
            cat_input[id_idx] = in_input
            cat_input[ood_idx] = out_input

            self.model.train()
            self.clsfier.train()       
            # cat_output = clsfier(model(cat_input))  # 192, 20
            # devices = list(range(torch.cuda.device_count()))
            temp_output = nn.parallel.data_parallel(self.model, cat_input, device_ids=list(range(NUM_DEVICE)))
            cat_output = nn.parallel.data_parallel(self.clsfier, temp_output, device_ids=list(range(NUM_DEVICE)))
            in_output = cat_output[id_idx] # torch.Size([64, 21])
            # in_output = cat_output[:in_len]
            # # train NN with sigmoid 
            in_conf = torch.sigmoid(in_output)[:,-1].mean()
            in_confs.update(in_conf.data, in_len)
            # in_loss = criterion(in_output, in_target.to(in_output.device)) 
            in_loss = criterion(in_output[:,:-1], in_target[:,:-1].to(in_output.device))   #in_target 64 * 21
            in_out_loss = criterion(in_output[:,-1], in_target[:,-1].to(in_output.device)) # in_target[:,-1]   [0]

            # out_output = cat_output[in_len:] # 128, 20
            out_output = cat_output[ood_idx] # torch.Size([64, 21])
            out_conf = torch.sigmoid(out_output)[:,-1].mean()
            out_confs.update(out_conf.data, out_len)
            out_loss = criterion(out_output, out_target.to(out_output.device).float())  # out_target 64 * 21 -> [0,0,...0,1]

            in_losses.update(in_loss.data, in_len) 
            in_out_losses.update(in_out_loss.data, in_len) 
            out_losses.update(out_loss.data, out_len)

            # loss = in_loss * 66/67 + self.args.gamma * in_out_loss * 1/67 + self.args.beta * out_loss
            loss = in_loss * 20/21 + self.args.gamma * in_out_loss * 1/21 + self.args.beta * out_loss
            # loss = in_loss + self.args.beta * out_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % self.args.print_freq == 0:
                log.debug('Epoch: [{0}] Batch#[{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'In Loss {in_loss.val:.4f} ({in_loss.avg:.4f})\t'
                    'In Out Loss {in_out_loss.val:.4f} ({in_out_loss.avg:.4f})\t'
                    'Out Loss {out_loss.val:.4f} ({out_loss.avg:.4f})\t'
                    'In Conf {in_confs.val:.4f} ({in_confs.avg:.4f})\t'
                    'OOD Conf {out_confs.val:.4f} ({out_confs.avg:.4f})'.format(
                        epoch, i, len(train_loader_in),
                        batch_time=batch_time,
                        in_loss=in_losses,  in_out_loss=in_out_losses, 
                        out_loss=out_losses, out_confs=out_confs,
                        in_confs=in_confs))
        
        self.push_to_cyclic_buffer(epoch_buffer_in, epoch_buffer_out, epoch)

        # log to TensorBoard
        # if self.args.tensorboard:
        #     log_value('nat_train_acc', nat_top1.avg, epoch)

    def train_blr_energy(self, train_loader_in, train_loader_out, criterion, optimizer, epoch, save_dir, log, bayes = False):
        """Train for one epoch on the training set"""
        batch_time = AverageMeter()

        in_losses = AverageMeter()
        in_energy_losses = AverageMeter()
        in_energy_all = AverageMeter()
        out_energy_all = AverageMeter()
        out_energy_losses = AverageMeter()
        log.debug("######## Start training NN at epoch {} ########".format(epoch) )

        end = time.time()
        if bayes:
            epoch_buffer_in = torch.empty(0, 3, 224, 224)
            epoch_buffer_out = torch.empty(0, 3, 224, 224)

        for i, (in_set, out_set) in enumerate(zip(train_loader_in, train_loader_out)):
            in_len = len(in_set[0])
            out_len = len(out_set[0])
            if bayes:
                epoch_buffer_in = torch.cat((epoch_buffer_in, in_set[0]), 0)
                epoch_buffer_out = torch.cat((epoch_buffer_out, out_set[0]), 0)
            in_input = in_set[0].cuda() # 64, 3, 244, 244
            in_target = in_set[1].cuda().float()
            out_input = out_set[0].to(in_input.device) # 128, 3, 244, 244

            # V1: concatenate then feed forward 
            # cat_input = torch.cat((in_input, out_input), 0)
            # NUM_DEVICE = 4
            # self.model.train()
            # self.clsfier.train()       
            # cat_output = nn.parallel.data_parallel(self.model, cat_input, device_ids=list(range(NUM_DEVICE)))
            # cat_output = nn.parallel.data_parallel(self.clsfier, cat_output, device_ids=list(range(NUM_DEVICE)))
            # cat_energy = -1 * torch.sum(torch.log(1+torch.exp(cat_output)), dim = 1)
            # in_output = cat_output[:in_len]
            # out_output = cat_output[in_len:]
            # in_energy = cat_energy[:in_len]
            # out_energy = cat_energy[in_len:]

            # V2: twice feed forward 
            NUM_DEVICE = 4
            self.model.train()
            self.clsfier.train()       
            in_output = nn.parallel.data_parallel(self.model, in_input, device_ids=list(range(NUM_DEVICE)))
            in_output = nn.parallel.data_parallel(self.clsfier, in_output, device_ids=list(range(NUM_DEVICE)))
            in_energy = -1 * torch.sum(torch.log(1+torch.exp(in_output)), dim = 1)

            out_output = nn.parallel.data_parallel(self.model, out_input, device_ids=list(range(NUM_DEVICE)))
            out_output = nn.parallel.data_parallel(self.clsfier, out_output, device_ids=list(range(NUM_DEVICE)))
            out_energy = -1 * torch.sum(torch.log(1+torch.exp(out_output)), dim = 1)

            # calculate loss
            in_loss = criterion(in_output, in_target.to(in_output.device)) 
            in_energy_loss = torch.pow(F.relu(in_energy-self.args.m_in), 2).mean() 
            out_energy_loss = torch.pow(F.relu(self.args.m_out-out_energy), 2).mean()
            loss = in_loss + self.args.eta * (in_energy_loss + out_energy_loss)

            # update Average meters
            in_losses.update(in_loss.data, in_len) 
            in_energy_losses.update(in_energy_loss.data, in_len)
            out_energy_losses.update(out_energy_loss.data, out_len)
            in_energy_all.update(in_energy.mean().data, in_len)
            out_energy_all.update(out_energy.mean().data, in_len)

            # update params
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % self.args.print_freq == 0:
                log.debug('Epoch: [{0}] Batch#[{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'In Ls {in_loss.val:.4f} ({in_loss.avg:.4f})\t'
                    'E_in Ls {in_energy_loss.val:.4f} ({in_energy_loss.avg:.4f})\t'
                    'E_out Ls {out_energy_loss.val:.4f} ({out_energy_loss.avg:.4f})\t'
                    'E_in Raw {in_energy_all.val:.4f} ({in_energy_all.avg:.4f})\t'
                    'E_out Raw {out_energy_all.val:.4f} ({out_energy_all.avg:.4f})\t'.format(
                        epoch, i, len(train_loader_in),
                        batch_time=batch_time,
                        in_loss=in_losses, 
                        in_energy_loss = in_energy_losses,
                        out_energy_loss=out_energy_losses, 
                        in_energy_all=in_energy_all,
                        out_energy_all=out_energy_all,
                        ))
        if bayes:
            self.push_to_cyclic_buffer(epoch_buffer_in, epoch_buffer_out, epoch)

        # log to TensorBoard
        # if self.args.tensorboard:
        #     log_value('nat_train_acc', nat_top1.avg, epoch)

    def train_blr_energy_single(self, train_loader_in, train_loader_out, criterion, optimizer, epoch, save_dir, log):
        """Train for one epoch on the training set"""
        batch_time = AverageMeter()

        in_losses = AverageMeter()
        in_energy_losses = AverageMeter()
        in_energy_all = AverageMeter()
        out_energy_all = AverageMeter()
        out_energy_losses = AverageMeter()
        log.debug("######## Start training NN at epoch {} ########".format(epoch) )

        end = time.time()

        # epoch_buffer_in = torch.empty(0, 3, 224, 224)
        # epoch_buffer_out = torch.empty(0, 3, 224, 224)

        for i, (in_set, out_set) in enumerate(zip(train_loader_in, train_loader_out)):
            in_len = len(in_set[0])
            out_len = len(out_set[0])

            in_input = in_set[0].cuda() # 64, 3, 244, 244
            in_target = in_set[1].cuda().float()
            out_input = out_set[0].to(in_input.device) # 128, 3, 244, 244

            cat_input = torch.cat((in_input, out_input), 0) # 192, 3, 244, 244

            self.model.train()
            self.clsfier.train()       
            # cat_output = self.clsfier(self.model(cat_input))  # 192, 20
            cat_output = self.model(cat_input)
            cat_output = self.clsfier(cat_output)
            cat_energy = -1 * torch.sum(torch.log(1+torch.exp(cat_output)), dim = 1)
            in_output = cat_output[:in_len]
            in_energy = cat_energy[:in_len]
            # print("in energy max: ", in_energy.max())
            # print("in energy min: ", in_energy.min())
            # print("in energy mean: ", in_energy.mean().data)
            # # train NN with sigmoid 
            # in_conf = torch.sigmoid(in_output).mean()
            #in_confs.update(in_conf.data, in_len)
            in_loss = criterion(in_output, in_target)   #in_target 64 * 21
            in_energy_loss = torch.pow(F.relu(in_energy-self.args.m_in), 2).mean() 

            # out_output = cat_output[in_len:] # 128, 20
            out_output = cat_output[in_len:] # torch.Size([64, 20])
            out_energy = cat_energy[in_len:]
            # out_conf = torch.sigmoid(out_output)[:,-1].mean()
            # out_confs.update(out_conf.data, out_len)
            # print("out energy max: ", out_energy.max())
            # print("out energy min: ", out_energy.min())
            # print("out energy mean: ", out_energy.mean().data)

            out_energy_loss = torch.pow(F.relu(self.args.m_out-out_energy), 2).mean()

            in_losses.update(in_loss.data, in_len) 
            in_energy_losses.update(in_energy_loss.data, in_len)
            in_energy_all.update(in_energy.mean().data, in_len)
            out_energy_all.update(out_energy.mean().data, in_len)
            out_energy_losses.update(out_energy_loss.data, out_len)

            loss = in_loss + self.args.eta * (in_energy_loss + out_energy_loss)
            # loss = in_loss + self.args.beta * out_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % self.args.print_freq == 0:
                log.debug('Epoch: [{0}] Batch#[{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'In Ls {in_loss.val:.4f} ({in_loss.avg:.4f})\t'
                    'E_in Ls {in_energy_loss.val:.4f} ({in_energy_loss.avg:.4f})\t'
                    'E_out Ls {out_energy_loss.val:.4f} ({out_energy_loss.avg:.4f})\t'
                    'E_in Raw {in_energy_all.val:.4f} ({in_energy_all.avg:.4f})\t'
                    'E_out Raw {out_energy_all.val:.4f} ({out_energy_all.avg:.4f})\t'.format(
                        epoch, i, len(train_loader_in),
                        batch_time=batch_time,
                        in_loss=in_losses, 
                        in_energy_loss = in_energy_losses,
                        out_energy_loss=out_energy_losses, 
                        in_energy_all=in_energy_all,
                        out_energy_all=out_energy_all,
                        ))
         
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
            log.debug("Before inverse. eigenvalue (ASC): {}".format(A_eig_val[:20]))
            log.debug("Before inverse. eigenvalue (DES): {}".format(A_eig_val[-20:]))
            A = A.detach().cpu().numpy()
            # inv = torch.inverse(A)
            inv = np.linalg.inv(A)
            inv = torch.from_numpy(inv).cuda()
            self.mu_w[0] = torch.matmul(inv, B).squeeze()
            temp_cov = self.sigma*inv        
            eig_val, eig_vec = torch.symeig(temp_cov, eigenvectors=True)
            log.debug("After inverse. eigenvalue (ASC): {}".format(eig_val[:20]) )
            log.debug("After inverse. eigenvalue (DES): {}".format(eig_val[-20:]) )
            if torch.any(eig_val < 0):
                log.debug("###### WARNING! NEGATIVE EIGENVALUE ######" )
                self.cov_w[0] = torch.matmul(torch.matmul(eig_vec, torch.diag(torch.abs(eig_val))), torch.t(eig_vec))
            else:
                self.cov_w[0] = temp_cov
            # print('singularity check: matrix not singular.')

    def validate(self, val_loader, epoch, log):
        in_cls_confs = AverageMeter()
        in_bys_confs = AverageMeter()
        self.model.eval()
        self.clsfier.eval()
        init = True
        log.debug("######## Start validation ########")
        # gts = {i:[] for i in range(0, num_classes)}
        # preds = {i:[] for i in range(0, num_classes)}
        with torch.no_grad():
            for i, (images, labels) in enumerate(val_loader):
                images = images.cuda()
                # extra_col = torch.zeros(len(labels), 1)
                # labels = torch.cat( (labels, extra_col), dim = 1).cuda().float()
                labels = labels.cuda().float()
                outputs = self.model(images)
                latent_z = self.clsfier.intermediate_forward(outputs) #DEBUG
                in_bys_conf = torch.sigmoid(torch.mm(self.beta_s, latent_z.T).T).mean()  #DEBUG
                outputs = self.clsfier(outputs)
                outputs = torch.sigmoid(outputs)
                pred = outputs[:, :-1].squeeze().data.cpu().numpy()
                in_cls_conf = outputs[:, -1].mean() # 64       #DEBUG        
                in_cls_confs.update(in_cls_conf.data, len(labels))  #DEBUG
                in_bys_confs.update(in_bys_conf.data, len(labels))  #DEBUG
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
                #DEBUG
                if i % self.args.print_freq == 0:
                    log.debug('Epoch: [{0}] Batch#[{1}/{2}]\t'
                        'In Classifer Conf {in_cls_confs.val:.4f} ({in_cls_confs.avg:.4f})\t'
                        'In Bayesian Conf {in_bys_confs.val:.4f} ({in_bys_confs.avg:.4f})'.format(
                            epoch, i, len(val_loader), in_bys_confs=in_bys_confs, in_cls_confs=in_cls_confs))

        FinalMAPs = []
        num_per_class = []
        # for j in range(1, self.num_classes):
        #     # precision, recall, thresholds = metrics.precision_recall_curve(all_ground_truth[i], all_pred[i])
        #     precision, recall, thresholds = metrics.precision_recall_curve(all_ground_truth[:, j], all_pred[:, j])
        #     if j not in [27, 30, 31, 32, 33, 34, 35, 36, 37, 38, 70, 78]:
        #         FinalMAPs.append(metrics.auc(recall, precision))
        #         print(f"class {j} auc: {metrics.auc(recall, precision)}")
        # print("final map: ", np.mean(FinalMAPs))
        for j in range(self.num_classes):
            # precision, recall, thresholds = metrics.precision_recall_curve(all_ground_truth[i], all_pred[i])
            precision, recall, thresholds = metrics.precision_recall_curve(all_ground_truth[:, j], all_pred[:, j])
            FinalMAPs.append(metrics.auc(recall, precision))
            print(f"class {j} auc: {metrics.auc(recall, precision)}")
        print("final map: ", np.mean(FinalMAPs))

        # # log to TensorBoard
        # if self.args.tensorboard:
        #     log_value('mAP', np.mean(FinalMAPs) epoch)

        return np.mean(FinalMAPs)

    def validate_random(self, val_loader, epoch, log):
        in_cls_confs = AverageMeter()
        self.model.eval()
        self.clsfier.eval()
        init = True
        log.debug("######## Start validation ########")
        # gts = {i:[] for i in range(0, num_classes)}
        # preds = {i:[] for i in range(0, num_classes)}
        with torch.no_grad():
            for i, (images, labels) in enumerate(val_loader):
                images = images.cuda()
                # extra_col = torch.zeros(len(labels), 1)
                # labels = torch.cat( (labels, extra_col), dim = 1).cuda().float()
                labels = labels.cuda().float()
                outputs = self.model(images)
                outputs = self.clsfier(outputs)
                outputs = torch.sigmoid(outputs)
                pred = outputs[:, :-1].squeeze().data.cpu().numpy()
                in_cls_conf = outputs[:, -1].mean() # 64       #DEBUG        
                in_cls_confs.update(in_cls_conf.data, len(labels))  #DEBUG
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
                #DEBUG
                if i % self.args.print_freq == 0:
                    log.debug('Epoch: [{0}] Batch#[{1}/{2}]\t'
                        'In Classifer Conf {in_cls_confs.val:.4f} ({in_cls_confs.avg:.4f})'.format(
                            epoch, i, len(val_loader), in_cls_confs=in_cls_confs))

        FinalMAPs = []
        num_per_class = []
        for j in range(self.num_classes):
            # precision, recall, thresholds = metrics.precision_recall_curve(all_ground_truth[i], all_pred[i])
            precision, recall, thresholds = metrics.precision_recall_curve(all_ground_truth[:, j], all_pred[:, j])
            # if j not in [27, 30, 31, 32, 33, 34, 35, 36, 37, 38, 70, 78]:
            FinalMAPs.append(metrics.auc(recall, precision))
            print(f"class {j} auc: {metrics.auc(recall, precision)}")
        print("final map: ", np.mean(FinalMAPs))

        return np.mean(FinalMAPs)

    def validate_energy(self, val_loader, epoch, log, bayes = False):
        in_energy = AverageMeter()
        in_bys_confs = AverageMeter()
        self.model.eval()
        self.clsfier.eval()
        init = True
        log.debug("######## Start validation ########")
        # gts = {i:[] for i in range(0, num_classes)}
        # preds = {i:[] for i in range(0, num_classes)}
        with torch.no_grad():
            for i, (images, labels) in enumerate(val_loader):
                images = images.cuda()
                # extra_col = torch.zeros(len(labels), 1)
                # labels = torch.cat( (labels, extra_col), dim = 1).cuda().float()
                labels = labels.cuda().float()
                outputs = self.model(images)
                if bayes: 
                    latent_z = self.clsfier.intermediate_forward(outputs) #DEBUG
                    in_bys_conf = torch.sigmoid(torch.mm(self.beta_s, latent_z.T).T).mean()  #DEBUG
                    in_bys_confs.update(in_bys_conf.data, len(labels))  #DEBUG
                    if i % self.args.print_freq == 0:
                        log.debug('Epoch: [{0}] Batch#[{1}/{2}]\t'
                                  'In Bayesian Conf {in_bys_confs.val:.4f} ({in_bys_confs.avg:.4f})'.format(
                                epoch, i, len(val_loader), in_bys_confs=in_bys_confs))
                outputs = self.clsfier(outputs)
                E_y = torch.log(1+torch.exp(outputs))
                energy = -1 * torch.sum(E_y, dim = 1).mean()
                in_energy.update(energy.mean().data, len(labels))  #DEBUG
                outputs = torch.sigmoid(outputs)  
                pred = outputs.squeeze().data.cpu().numpy()    
                # in_cls_confs.update(in_cls_conf.data, len(labels))  #DEBUG
                ground_truth = labels.squeeze().data.cpu().numpy()
                if init:
                    all_ground_truth = ground_truth 
                    all_pred = pred
                    init = False
                else:
                    all_ground_truth = np.vstack((all_ground_truth, ground_truth) )
                    all_pred = np.vstack((all_pred, pred))
    
                #DEBUG
                if i % self.args.print_freq == 0:
                    log.debug('Epoch: [{0}] Batch#[{1}/{2}]\t'
                        'In Classifer Energy {in_energy.val:.4f} ({in_energy.avg:.4f})'.format(
                            epoch, i, len(val_loader), in_energy=in_energy))

        FinalMAPs = []
        for j in range(self.num_classes):
            # precision, recall, thresholds = metrics.precision_recall_curve(all_ground_truth[i], all_pred[i])
            precision, recall, thresholds = metrics.precision_recall_curve(all_ground_truth[:, j], all_pred[:, j])
            # if j not in [27, 30, 31, 32, 33, 34, 35, 36, 37, 38, 70, 78]:
            FinalMAPs.append(metrics.auc(recall, precision))
            print(f"class {j} auc: {metrics.auc(recall, precision)}")
        print("final map: ", np.mean(FinalMAPs))

        return np.mean(FinalMAPs)


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

# def accuracy(output, target, topk=(1,)):
#     """Computes the precision@k for the specified values of k"""
#     maxk = max(topk)
#     batch_size = target.size(0)

#     _, pred = output.topk(maxk, 1, True, True)
#     pred = pred.t().to(target.device)
#     correct = pred.eq(target.view(1, -1).expand_as(pred))

#     res = []
#     for k in topk:
#         correct_k = correct[:k].view(-1).float().sum(0)
#         res.append(correct_k.mul_(100.0 / batch_size))
#     return res