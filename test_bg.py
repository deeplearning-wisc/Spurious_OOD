import argparse
import os
import logging

import numpy as np
import torch
import torchvision
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from models import Resnet
from datasets.color_mnist import get_biased_mnist_dataloader
from datasets.cub_dataset import WaterbirdDataset
from datasets.celebA_dataset import get_celebA_dataloader, get_celebA_ood_dataloader
from utils import AverageMeter, accuracy

parser = argparse.ArgumentParser(description='OOD Detection Evaluation based on Energy-score')

parser.add_argument('--exp-name', required = True, type=str, help='help identify checkpoint')
parser.add_argument('--name', '-n', required = True, type=str, help='name of experiment')
parser.add_argument('--in-dataset', default="color_mnist", type=str, help='name of the in-distribution dataset')
parser.add_argument('--model-arch', default='resnet18', type=str, help='model architecture e.g. resnet18')
parser.add_argument('--method', default='dann', type=str, help='method used for model training')
parser.add_argument('--print-freq', '-p', default=10, type=int, help='print frequency (default: 10)') 
parser.add_argument('--domain-num', default=2, type=int,
                    help='the number of environments for model training')
parser.add_argument('-b', '--batch-size', default= 64, type=int,
                    help='mini-batch size (default: 64) used for training id and ood')
parser.add_argument('--num-classes', default=2, type=int,
                    help='number of classes for model training')
parser.add_argument('--ood-batch-size', default= 64, type=int,
                    help='mini-batch size (default: 400) used for testing')
parser.add_argument('--data_label_correlation', default= 0.4, type=float,
                    help='data_label_correlation')
parser.add_argument('--test_epochs', "-e", default = "10", type=str,
                     help='# epoch to test performance')
parser.add_argument('--log_name',
                    help='Name of the Log File', type = str, default = "info_val.log")
parser.add_argument('--base-dir', default='output/ood_scores', type=str, help='result directory')
parser.add_argument('--gpu-ids', default='5', type=str,
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

# distributed training will be implemented in the updated version
# args.n_gpus =torch.cuda.device_count()
# if args.n_gpus > 1:
#     import torch.distributed as dist
#     import torch.multiprocessing as mp
#     from torch.utils.data.distributed import DistributedSampler
#     from torch.nn.parallel import DistributedDataParallel as DDP
#     args.multi_gpu = True
#     torch.distributed.init_process_group(
#         'nccl',
#         init_method='env://',
#         world_size=args.n_gpus,
#         rank=args.local_rank,
#     )
# else:
#     args.multi_gpu = False

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


    if args.in_dataset == "color_mnist":
        val_loader = get_biased_mnist_dataloader(args, root = './datasets/MNIST', batch_size=args.batch_size,
                                            data_label_correlation= args.data_label_correlation,
                                            n_confusing_labels= args.num_classes - 1,
                                            train=False, partial=True, cmap = "1")
        num_classes = args.num_classes
    elif args.in_dataset == "waterbird":
        val_dataset = WaterbirdDataset(data_correlation=args.data_label_correlation, split='test')
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
        num_classes = args.num_classes
    elif args.in_dataset == "celebA":
        val_loader = get_celebA_dataloader(args, split='test')
        num_classes = args.num_classes

    # create model
    model = Resnet(n_classes=args.num_classes, model=args.model_arch, method=args.method, domain_num=args.domain_num)
    model = model.cuda()

    test_epochs = args.test_epochs.split()
    if args.in_dataset == 'color_mnist':
        out_datasets = ['partial_color_mnist_0&1']
        #out_datasets = ['dtd', 'iSUN', 'LSUN_resize']
    elif args.in_dataset == 'waterbird':
        out_datasets = ['placesbg']
    elif args.in_dataset == 'color_mnist_multi':
        out_datasets = ['partial_color_mnist_0&1']
    elif args.in_dataset == 'celebA':
        out_datasets = ['celebA_ood']

    for test_epoch in test_epochs:
        cpts_directory = "/nobackup/spurious_ood/checkpoints/{in_dataset}/{name}/{exp}".format(in_dataset=args.in_dataset, name=args.name, exp=args.exp_name)
        cpts_dir = os.path.join(cpts_directory, "checkpoint_{epochs}.pth.tar".format(epochs=test_epoch))
        checkpoint = torch.load(cpts_dir)
        model.load_state_dict(checkpoint['state_dict_model'])
        model.eval()
        model.cuda()
        save_dir =  f"./energy_results/{args.in_dataset}/{args.name}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        print("processing ID dataset")
        id_energy  = get_id_energy(args, model, val_loader, test_epoch, log, method=args.method)
        with open(os.path.join(save_dir, f'energy_score_at_epoch_{test_epoch}.npy'), 'wb') as f:
            np.save(f, id_energy)
        for out_dataset in out_datasets:
            print("processing OOD dataset ", out_dataset)
            testloaderOut = get_ood_loader(out_dataset)
            ood_energy = get_ood_energy(args, model, testloaderOut, test_epoch, log, method=args.method)
            with open(os.path.join(save_dir, f'energy_score_{out_dataset}_at_epoch_{test_epoch}.npy'), 'wb') as f:
                np.save(f, ood_energy)

def get_ood_energy(args, model, val_loader, epoch, log, method):
    in_energy = AverageMeter()
    model.eval()
    init = True
    log.debug("######## Start collecting energy score ########")
    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            images = images.cuda()
            if method == "dann":
                _, outputs, _ = model(images, alpha=0)
            else:
                _, outputs = model(images)
            e_s = -torch.logsumexp(outputs, dim=1)
            e_s = e_s.data.cpu().numpy() 
            in_energy.update(e_s.mean(), len(labels))
            if init:
                sum_energy = e_s
                init = False
            else:
                sum_energy = np.concatenate((sum_energy, e_s))
            if i % args.print_freq == 0: 
                log.debug('Epoch: [{0}] Batch#[{1}/{2}]\t'
                    'Energy Sum {in_energy.val:.4f} ({in_energy.avg:.4f})'.format(
                        epoch, i, len(val_loader), in_energy=in_energy))
        return sum_energy

def get_id_energy(args, model, val_loader, epoch, log, method):
    in_energy = AverageMeter()
    top1 = AverageMeter()
    model.eval()
    init = True
    log.debug("######## Start collecting energy score ########")
    with torch.no_grad():
        all_preds = torch.tensor([])
        all_targets = torch.tensor([])
        for i, (images, labels, _) in enumerate(val_loader):
            images = images.cuda()
            if method == "dann":
                _, outputs, _ = model(images, alpha=0)
            else:
                _, outputs = model(images)
            all_targets = torch.cat((all_targets, labels),dim=0)
            all_preds = torch.cat((all_preds, outputs.argmax(dim=1).cpu()),dim=0)
            prec1 = accuracy(outputs.cpu().data, labels, topk=(1,))[0]
            top1.update(prec1, images.size(0))
            e_s = -torch.logsumexp(outputs, dim=1)
            e_s = e_s.data.cpu().numpy() 
            in_energy.update(e_s.mean(), len(labels)) 
            if init:
                sum_energy = e_s
                init = False
            else:
                sum_energy = np.concatenate((sum_energy, e_s))
            if i % args.print_freq == 0: 
                log.debug('Epoch: [{0}] Batch#[{1}/{2}]\t'
                    'Energy Sum {in_energy.val:.4f} ({in_energy.avg:.4f})'.format(
                        epoch, i, len(val_loader), in_energy=in_energy))
        log.debug(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
        return sum_energy

def get_ood_loader(out_dataset):
        mnist_transform = transforms.Compose([
                transforms.Resize(28),
                 transforms.CenterCrop(28),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5),
                             std=(0.5, 0.5, 0.5))])
        if out_dataset == 'dtd':
            testsetout = torchvision.datasets.ImageFolder(root="datasets/ood_datasets/dtd/images",
                                        transform=mnist_transform)
            testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=args.ood_batch_size, shuffle=True,
                                                     num_workers=2)
        elif 'partial_color_mnist' in out_dataset:
            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5),
                             std=(0.5, 0.5, 0.5))])
            testsetout = torchvision.datasets.ImageFolder("datasets/ood_datasets/{}".format(out_dataset),
                                        transform=test_transform)
            testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=args.ood_batch_size,
                                             shuffle=True, num_workers=2)
        elif 'placesbg' in out_dataset:
            scale = 256.0/224.0
            target_resolution = (224, 224)
            celeba_transform = transforms.Compose([
                transforms.Resize((int(target_resolution[0]*scale), int(target_resolution[1]*scale))),
                transforms.CenterCrop(target_resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            testsetout = torchvision.datasets.ImageFolder("datasets/ood_datasets/{}".format(out_dataset),
                                        transform=celeba_transform)
            subset = torch.utils.data.Subset(testsetout, np.random.choice(len(testsetout), 5000, replace=False))
            testloaderOut = torch.utils.data.DataLoader(subset, batch_size=args.ood_batch_size,
                                             shuffle=True, num_workers=2)           
        elif 'celebA_ood' in out_dataset:
            testloaderOut = get_celebA_ood_dataloader(args)
        else: # for iSUN and LSUN_Resize dataset, which are used for ColorMNIST
            testsetout = torchvision.datasets.ImageFolder("datasets/ood_datasets/{}".format(out_dataset),
                                        transform= mnist_transform)
            testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=args.ood_batch_size,
                                             shuffle=True, num_workers=2)
        return testloaderOut


if __name__ == '__main__':
    main()

