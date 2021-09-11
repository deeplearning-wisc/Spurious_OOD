import argparse
import os
import logging

import numpy as np
import torch
import torchvision
import torch.nn.parallel
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm_notebook as tqdm
import random 
import calculate_log as callog
from models import Resnet
from datasets.color_mnist import get_biased_mnist_dataloader
from datasets.cub_dataset import WaterbirdDataset
from datasets.celebA_dataset import get_celebA_dataloader, get_celebA_ood_dataloader, celebAOodDataset
from utils import AverageMeter, accuracy
import utils.svhn_loader as svhn
from datasets.gaussian_dataset import GaussianDataset
parser = argparse.ArgumentParser(description='OOD Detection Evaluation based on Energy-score')

# parser.add_argument('--exp-name', required = True, type=str, help='help identify checkpoint')
# parser.add_argument('--name', '-n', required = True, type=str, help='name of experiment')
parser.add_argument('--exp-name', default = 'erm_new_pretrain_1', type=str, help='help identify checkpoint')
parser.add_argument('--name', '-n', default = 'erm_rebuttal', type=str, help='name of experiment')
parser.add_argument('--in-dataset', default="color_mnist", type=str, help='name of the in-distribution dataset')
parser.add_argument('--model-arch', default='resnet18', type=str, help='model architecture e.g. resnet18')
parser.add_argument('--method', default='erm', type=str, help='method used for model training')
parser.add_argument('--print-freq', '-p', default=10, type=int, help='print frequency (default: 10)') 
parser.add_argument('--domain-num', default=4, type=int,
                    help='the number of environments for model training')
parser.add_argument('-b', '--batch-size', default= 1, type=int,
                    help='mini-batch size (default: 64) used for training id and ood')
parser.add_argument('--num-classes', default=2, type=int,
                    help='number of classes for model training')
parser.add_argument('--ood-batch-size', default= 1, type=int,
                    help='mini-batch size (default: 400) used for testing')
parser.add_argument('--data_label_correlation', default= 0.45, type=float,
                    help='data_label_correlation')
parser.add_argument('--test_epochs', "-e", default = "20", type=str,
                     help='# epoch to test performance')
parser.add_argument('--log_name',
                    help='Name of the Log File', type = str, default = "info_val.log")
parser.add_argument('--base-dir', default='output/ood_scores', type=str, help='result directory')
parser.add_argument('--gpu-ids', default='7', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--multi-gpu', default=False, type=bool)
parser.add_argument('--local_rank', default=-1, type=int,
                        help='rank for the current node')

args = parser.parse_args()

state = {k: v for k, v in args._get_kwargs()}
directory = "checkpoints/{in_dataset}/{name}/".format(in_dataset=args.in_dataset, name=args.name)
if not os.path.exists(directory):
    os.makedirs(directory)
save_state_file = os.path.join(directory, 'test_args.txt')
fw = open(save_state_file, 'w')
print(state, file=fw)
fw.close()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
if torch.cuda.is_available():
    torch.cuda.set_device(args.local_rank)
device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
        # self.shortcut = nn.Sequential()
        # if stride != 1 or in_planes != self.expansion*planes:
        #     self.shortcut = nn.Sequential(
        #         nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
        #         nn.BatchNorm2d(self.expansion*planes)
        #     )
    

    def forward(self, x):
        t = self.conv1(x)
        out = F.relu(self.bn1(t))
        model.record(t)
        model.record(out)
        t = self.conv2(out)
        out = self.bn2(self.conv2(out))
        model.record(t)
        model.record(out)
        t = self.downsample(x)
        # t = self.shortcut(x)
        out += t
        model.record(t)
        out = F.relu(out)
        model.record(out)
        
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=2):
        super(ResNet, self).__init__()
        self.in_planes = 64

        # self.conv1 = conv3x3(3,64)
        # self.bn1 = nn.BatchNorm2d(64)

        # for large input size
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
                        
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # end 
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.collecting = False
    
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out= self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # out = F.avg_pool2d(out, 4)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        y = self.linear(out)
        return y
    
    def record(self, t):
        if self.collecting:
            self.gram_feats.append(t)
    
    def gram_feature_list(self,x):
        self.collecting = True
        self.gram_feats = []
        with torch.no_grad():
            self.forward(x)
        self.collecting = False
        temp = self.gram_feats
        self.gram_feats = []
        return temp
    
    def load(self, path="resnet_svhn.pth"):
        tm = torch.load(path, map_location="cpu")        
        self.load_state_dict(tm)
    
    def get_min_max(self, data, power):
        mins = []
        maxs = []
        
        for i in range(0,len(data),128):
            batch = data[i:i+128].cuda()
            feat_list = self.gram_feature_list(batch)
            for L,feat_L in enumerate(feat_list):
                if L==len(mins):
                    mins.append([None]*len(power))
                    maxs.append([None]*len(power))
                
                for p,P in enumerate(power):
                    g_p = G_p(feat_L,P)
                    
                    current_min = g_p.min(dim=0,keepdim=True)[0]
                    current_max = g_p.max(dim=0,keepdim=True)[0]
                    
                    if mins[L][p] is None:
                        mins[L][p] = current_min
                        maxs[L][p] = current_max
                    else:
                        mins[L][p] = torch.min(current_min,mins[L][p])
                        maxs[L][p] = torch.max(current_max,maxs[L][p])
        
        return mins,maxs
    
    def get_deviations(self,data,power,mins,maxs):
        deviations = []
        
        for i in range(0,len(data),128):            
            batch = data[i:i+128].cuda()
            feat_list = self.gram_feature_list(batch)
            batch_deviations = []
            for L,feat_L in enumerate(feat_list):
                dev = 0
                for p,P in enumerate(power):
                    g_p = G_p(feat_L,P)
                    
                    dev +=  (F.relu(mins[L][p]-g_p)/torch.abs(mins[L][p]+10**-6)).sum(dim=1,keepdim=True)
                    dev +=  (F.relu(g_p-maxs[L][p])/torch.abs(maxs[L][p]+10**-6)).sum(dim=1,keepdim=True)
                batch_deviations.append(dev.cpu().detach().numpy())
            batch_deviations = np.concatenate(batch_deviations,axis=1)
            deviations.append(batch_deviations)
        deviations = np.concatenate(deviations,axis=0)
        
        return deviations



def detect(all_test_deviations,all_ood_deviations, verbose=True, normalize=True):
    average_results = {}
    for i in range(1,11):
        random.seed(i)
        
        validation_indices = random.sample(range(len(all_test_deviations)),int(0.1*len(all_test_deviations)))
        test_indices = sorted(list(set(range(len(all_test_deviations)))-set(validation_indices)))

        validation = all_test_deviations[validation_indices]
        test_deviations = all_test_deviations[test_indices]

        t95 = validation.mean(axis=0)+10**-7
        if not normalize:
            t95 = np.ones_like(t95)
        test_deviations = (test_deviations/t95[np.newaxis,:]).sum(axis=1)
        ood_deviations = (all_ood_deviations/t95[np.newaxis,:]).sum(axis=1)
        
        results = callog.compute_metric(-test_deviations,-ood_deviations)
        for m in results:
            average_results[m] = average_results.get(m,0)+results[m]
    
    for m in average_results:
        average_results[m] /= i
    if verbose:
        callog.print_results(average_results)
    return average_results

def cpu(ob):
    for i in range(len(ob)):
        for j in range(len(ob[i])):
            ob[i][j] = ob[i][j].cpu()
    return ob

def cuda(ob):
    for i in range(len(ob)):
        for j in range(len(ob[i])):
            ob[i][j] = ob[i][j].cuda()
    return ob

class Detector:
    def __init__(self):
        self.all_test_deviations = None
        self.mins = {}
        self.maxs = {}
        
        self.classes = range(2)
    
    def compute_minmaxs(self,data_train,POWERS=[10]):
        for PRED in tqdm(self.classes):
            train_indices = np.where(np.array(train_preds)==PRED)[0]
            train_PRED = torch.squeeze(torch.stack([data_train[i][0] for i in train_indices]),dim=1)
            mins,maxs = model.get_min_max(train_PRED,power=POWERS)
            self.mins[PRED] = cpu(mins)
            self.maxs[PRED] = cpu(maxs)
            torch.cuda.empty_cache()
    
    def compute_test_deviations(self,POWERS=[10]):
        all_test_deviations = None
        test_classes = []
        for PRED in tqdm(self.classes):
            test_indices = np.where(np.array(test_preds)==PRED)[0]
            test_PRED = torch.squeeze(torch.stack([data[i][0] for i in test_indices]),dim=1)
            test_confs_PRED = np.array([test_confs[i] for i in test_indices])
            
            test_classes.extend([PRED]*len(test_indices))
            
            mins = cuda(self.mins[PRED])
            maxs = cuda(self.maxs[PRED])
            test_deviations = model.get_deviations(test_PRED,power=POWERS,mins=mins,maxs=maxs)/test_confs_PRED[:,np.newaxis]
            cpu(mins)
            cpu(maxs)
            if all_test_deviations is None:
                all_test_deviations = test_deviations
            else:
                all_test_deviations = np.concatenate([all_test_deviations,test_deviations],axis=0)
            torch.cuda.empty_cache()
        self.all_test_deviations = all_test_deviations
        
        self.test_classes = np.array(test_classes)
    
    def compute_ood_deviations(self,ood,POWERS=[10]):
        ood_preds = []
        ood_confs = []
        
        for idx in range(0,len(ood),128):
            batch = torch.squeeze(torch.stack([x[0] for x in ood[idx:idx+128]]),dim=1).cuda()
            logits = model(batch)
            confs = F.softmax(logits,dim=1).cpu().detach().numpy()
            preds = np.argmax(confs,axis=1)
            
            ood_confs.extend(np.max(confs,axis=1))
            ood_preds.extend(preds)  
            torch.cuda.empty_cache()
        print("Done")
        
        ood_classes = []
        all_ood_deviations = None
        for PRED in tqdm(self.classes):
            ood_indices = np.where(np.array(ood_preds)==PRED)[0]
            if len(ood_indices)==0:
                continue
            ood_classes.extend([PRED]*len(ood_indices))
            
            ood_PRED = torch.squeeze(torch.stack([ood[i][0] for i in ood_indices]),dim=1)
            ood_confs_PRED =  np.array([ood_confs[i] for i in ood_indices])
            mins = cuda(self.mins[PRED])
            maxs = cuda(self.maxs[PRED])
            ood_deviations = model.get_deviations(ood_PRED,power=POWERS,mins=mins,maxs=maxs)/ood_confs_PRED[:,np.newaxis]
            cpu(self.mins[PRED])
            cpu(self.maxs[PRED])            
            if all_ood_deviations is None:
                all_ood_deviations = ood_deviations
            else:
                all_ood_deviations = np.concatenate([all_ood_deviations,ood_deviations],axis=0)
            torch.cuda.empty_cache()
            
        self.ood_classes = np.array(ood_classes)
        
        average_results = detect(self.all_test_deviations,all_ood_deviations)
        return average_results, self.all_test_deviations, all_ood_deviations

def G_p(ob, p):
    temp = ob.detach()
    
    temp = temp**p
    temp = temp.reshape(temp.shape[0],temp.shape[1],-1)
    temp = ((torch.matmul(temp,temp.transpose(dim0=2,dim1=1)))).sum(dim=2) 
    temp = (temp.sign()*torch.abs(temp)**(1/p)).reshape(temp.shape[0],-1)
    
    return temp

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



if __name__ == "__main__":
    log = logging.getLogger(__name__)
    formatter = logging.Formatter('%(asctime)s : %(message)s')
    fileHandler = logging.FileHandler(os.path.join(directory, args.log_name), mode='w')
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)
    log.setLevel(logging.DEBUG)
    log.addHandler(fileHandler)
    log.addHandler(streamHandler) 


    if args.in_dataset == "color_mnist":
        val_loader = get_biased_mnist_dataloader(args, root = './datasets/MNIST', batch_size=args.batch_size,
                                            data_label_correlation= args.data_label_correlation,
                                            n_confusing_labels= args.num_classes - 1,
                                            train=False, partial=True, cmap = "1")
        train_loader = get_biased_mnist_dataloader(args, root = './datasets/MNIST', batch_size=args.batch_size,
                                            data_label_correlation= args.data_label_correlation,
                                            n_confusing_labels= args.num_classes - 1,
                                            train=True, partial=True, cmap = "1")
        num_classes = args.num_classes
    elif args.in_dataset == "waterbird":
        train_dataset = WaterbirdDataset(data_correlation=args.data_label_correlation, split='train')
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_dataset = WaterbirdDataset(data_correlation=args.data_label_correlation, split='test')
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
        num_classes = args.num_classes
    elif args.in_dataset == "celebA":
        train_loader = get_celebA_dataloader(args, split="train")
        val_loader = get_celebA_dataloader(args, split='test')
        num_classes = args.num_classes

    # create model
    # model = Resnet(n_classes=args.num_classes, model=args.model_arch, method=args.method, domain_num=args.domain_num)
    model = ResNet(BasicBlock, [2,2,2,2], num_classes=2)
    model = model.cuda()

    test_epochs = args.test_epochs.split()
    if args.in_dataset == 'color_mnist':
        out_datasets = ['partial_color_mnist_0&1', 'gaussian', 'dtd', 'iSUN', 'LSUN_resize']
        #out_datasets = ['dtd', 'iSUN', 'LSUN_resize']
    elif args.in_dataset == 'waterbird':
        # out_datasets = ['placesbg']
        out_datasets = [ 'gaussian', 'placesbg', 'SVHN', 'iSUN', 'LSUN_resize']
    elif args.in_dataset == 'celebA':
        out_datasets = ['celebA_ood', 'gaussian', 'SVHN', 'iSUN', 'LSUN_resize']

    if args.in_dataset == 'color_mnist':
        cpts_directory = "./checkpoints/{in_dataset}/{name}/{exp}".format(in_dataset=args.in_dataset, name=args.name, exp=args.exp_name)
    else:
        cpts_directory = "./checkpoints/{in_dataset}/{name}/{exp}".format(in_dataset=args.in_dataset, name=args.name, exp=args.exp_name)
        # cpts_directory = "/nobackup-slow/spurious_ood/checkpoints/{in_dataset}/{name}/{exp}".format(in_dataset=args.in_dataset, name=args.name, exp=args.exp_name)
    
    for test_epoch in test_epochs:
        cpts_dir = os.path.join(cpts_directory, "checkpoint_{epochs}.pth.tar".format(epochs=test_epoch))
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
        save_dir =  f"./energy_results/{args.in_dataset}/{args.name}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        print("processing ID dataset")

        data_train = list(train_loader)
        data = list(val_loader)

        correct = 0
        total = 0
        for x,y,_ in val_loader:
            x = x.cuda()
            y = y.numpy()
            correct += (y==np.argmax(model(x).detach().cpu().numpy(),axis=1)).sum()
            total += y.shape[0]
        print("Accuracy: ",correct/total)
        
        train_preds = []
        train_confs = []
        train_logits = []
        for idx in range(0,len(data_train),128):
            batch = torch.squeeze(torch.stack([x[0] for x in data_train[idx:idx+128]]),dim=1).cuda()
            
            logits = model(batch)
            confs = F.softmax(logits,dim=1).cpu().detach().numpy()
            preds = np.argmax(confs,axis=1)
            logits = (logits.cpu().detach().numpy())

            train_confs.extend(np.max(confs,axis=1))    
            train_preds.extend(preds)
            train_logits.extend(logits)
        print("Done")

        test_preds = []
        test_confs = []
        test_logits = []

        for idx in range(0,len(data),128):
            batch = torch.squeeze(torch.stack([x[0] for x in data[idx:idx+128]]),dim=1).cuda()
            
            logits = model(batch)
            confs = F.softmax(logits,dim=1).cpu().detach().numpy()
            preds = np.argmax(confs,axis=1)
            logits = (logits.cpu().detach().numpy())

            test_confs.extend(np.max(confs,axis=1))    
            test_preds.extend(preds)
            test_logits.extend(logits)
        print("Done")

        detector = Detector()
        detector.compute_minmaxs(data_train,POWERS=range(1,11))

        detector.compute_test_deviations(POWERS=range(1,11))

        for out_dataset in out_datasets:
            print("processing OOD dataset ", out_dataset)
            ood_data = list(get_ood_loader(out_dataset, args.in_dataset))
            detector.compute_ood_deviations(ood_data,POWERS=range(1,11))
        

        # id_energy  = get_id_energy(args, model, val_loader, test_epoch, log, method=args.method)
        # with open(os.path.join(save_dir, f'energy_score_at_epoch_{test_epoch}.npy'), 'wb') as f:
        #     np.save(f, id_energy)
        # for out_dataset in out_datasets:
        #     print("processing OOD dataset ", out_dataset)
        #     testloaderOut = get_ood_loader(out_dataset, args.in_dataset)
        #     ood_energy = get_ood_energy(args, model, testloaderOut, test_epoch, log, method=args.method)
        #     with open(os.path.join(save_dir, f'energy_score_{out_dataset}_at_epoch_{test_epoch}.npy'), 'wb') as f:
        #         np.save(f, ood_energy)
