import os
import torch
import numpy as np
import pandas as pd
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class celebADataset(Dataset):
    def __init__(self, train=True):
        self.split_dict = {
            'train': 0,
            'validation': 1,
            'test': 2
        }
        self.env_dict = {
            (0, 0): 0,
            (0, 1): 1,
            (1, 0): 2,
            (1, 1): 3
        }
        self.train = train
        self.dataset_name = 'celebA'
        self.dataset_dir = os.path.join("/nobackup/spurious_ood", self.dataset_name)
        if not os.path.exists(self.dataset_dir):
            raise ValueError(
                f'{self.dataset_dir} does not exist yet. Please generate the dataset first.')
        self.metadata_df = pd.read_csv(
            os.path.join(self.dataset_dir, 'celebA_split.csv'))
        if self.train:
            self.metadata_df = self.metadata_df[self.metadata_df['split']==self.split_dict['train']]
        else:
            self.metadata_df = self.metadata_df[self.metadata_df['split']==self.split_dict['test']]

        self.y_array = self.metadata_df['Gray_Hair'].values
        self.gender_array = self.metadata_df['Male'].values
        self.filename_array = self.metadata_df['image_id'].values
        self.transform = get_transform_cub(self.train)

    def __len__(self):
        return len(self.filename_array)

    def __getitem__(self, idx):
        y = self.y_array[idx]
        gender = self.gender_array[idx]
        img_filename = os.path.join(
            self.dataset_dir,
            'img_align_celeba',
            self.filename_array[idx])
        img = Image.open(img_filename).convert('RGB')
        img = self.transform(img)

        return img, y, self.env_dict[(y, gender)]


class celebAOodDataset(Dataset):
    def __init__(self):
        self.dataset_name = 'celebA'
        self.dataset_dir = os.path.join("/nobackup/spurious_ood", self.dataset_name)
        if not os.path.exists(self.dataset_dir):
            raise ValueError(
                f'{self.dataset_dir} does not exist yet. Please generate the dataset first.')
        self.metadata_df = pd.read_csv(
            os.path.join(self.dataset_dir, 'celebA_ood.csv'))

        self.filename_array = self.metadata_df['image_id'].values
        self.transform = get_transform_cub(train=False)

    def __len__(self):
        return len(self.filename_array)

    def __getitem__(self, idx):
        img_filename = os.path.join(
            self.dataset_dir,
            'img_align_celeba',
            self.filename_array[idx])
        img = Image.open(img_filename).convert('RGB')
        img = self.transform(img)

        return img, img


def get_transform_cub(train):
    orig_w = 178
    orig_h = 218
    orig_min_dim = min(orig_w, orig_h)
    target_resolution = (224, 224)

    if not train:
        transform = transforms.Compose([
            transforms.CenterCrop(orig_min_dim),
            transforms.Resize(target_resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        # Orig aspect ratio is 0.81, so we don't squish it in that direction any more
        transform = transforms.Compose([
            transforms.RandomResizedCrop(
                target_resolution,
                scale=(0.7, 1.0),
                ratio=(1.0, 1.3333333333333333),
                interpolation=2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    return transform


def get_celebA_dataloader(args, train=True):
    kwargs = {'pin_memory': False, 'num_workers': 8, 'drop_last': True}
    dataset = celebADataset(train=train)
    if args.multi_gpu:
            ddp_sampler = DistributedSampler(dataset, num_replicas=args.n_gpus, rank=args.local_rank)
            dataloader = DataLoader(dataset=dataset,
                                    batch_size=args.batch_size,
                                    sampler=ddp_sampler,
                                    shuffle=False,
                                    **kwargs)
    else:
        dataloader = DataLoader(dataset=dataset,
                                    batch_size=args.batch_size,
                                    shuffle=True,
                                    **kwargs)
    return dataloader


def get_celebA_ood_dataloader(args):
    kwargs = {'pin_memory': False, 'num_workers': 8, 'drop_last': True}
    dataset = celebAOodDataset()
    dataloader = DataLoader(dataset=dataset,
                                batch_size=args.ood_batch_size,
                                shuffle=True,
                                **kwargs)
    return dataloader


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='OOD training for multi-label classification')
    parser.add_argument('-b', '--batch-size', default=64, type=int,
                    help='mini-batch size (default: 64) used for training')
    parser.add_argument('--ood-batch-size', default= 64, type=int,
                    help='mini-batch size (default: 400) used for testing')
    parser.add_argument('--multi-gpu', default=False, type=bool)
    args = parser.parse_args()

    dataloader = get_celebA_dataloader(args, train=True)
    ood_dataloader = get_celebA_ood_dataloader(args)