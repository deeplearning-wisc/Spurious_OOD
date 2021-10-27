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
    def __init__(self, args, split):
        self.split_dict = {
            'train': 0,
            'val': 1,
            'test': 2
        }
        # (y, gender)
        self.env_dict = {
            (0, 0): 0,   # nongrey hair, female
            (0, 1): 1,   # nongrey hair, male
            (1, 0): 2,   # gray hair, female
            (1, 1): 3    # gray hair, male
        }
        self.split = split
        self.dataset_name = 'celebA'
        # self.dataset_dir = os.path.join("datasets/", self.dataset_name)
        self.dataset_dir = '/nobackup/sonic/celebA'
        if not os.path.exists(self.dataset_dir):
            raise ValueError(
                f'{self.dataset_dir} does not exist yet. Please generate the dataset first.')
        self.metadata_df = pd.read_csv(
            os.path.join(self.dataset_dir, 'celebA_split.csv'))
        self.metadata_df = self.metadata_df[self.metadata_df['split']==self.split_dict[self.split]]

        self.y_array = self.metadata_df['Gray_Hair'].values
        self.gender_array = self.metadata_df['Male'].values
        self.filename_array = self.metadata_df['image_id'].values
        self.transform = get_transform_cub(self.split=='train')
        if self.split == 'train':
            self.subsample(args.data_label_correlation)
    
    def subsample(self, ratio = 0.6):
        np.random.seed(1)
        train_group_idx = {
            (0, 0): np.array([]),   # nongrey hair, female
            (0, 1): np.array([]),   # nongrey hair, male
            (1, 0): np.array([]),   # gray hair, female
            (1, 1): np.array([])    # gray hair, male
        }
        for idx, (y, gender) in enumerate(zip(self.y_array, self.gender_array)):
            train_group_idx[(y, gender)] = np.append(train_group_idx[(y, gender)],idx)
        sample_size = int(ratio/(1-ratio)*len(train_group_idx[(1, 0)]))
        undersampled_idx_00 = np.random.choice(train_group_idx[(0, 0)], sample_size, replace = False)
        undersampled_idx_11 = np.random.choice(train_group_idx[(1, 1)], sample_size, replace = False)
        undersampled_idx = np.concatenate( (train_group_idx[(1, 0)], undersampled_idx_00, undersampled_idx_11, train_group_idx[(0, 1)]) )
        undersampled_idx = undersampled_idx.astype(int)
        self.y_array = self.y_array[undersampled_idx]
        self.gender_array = self.gender_array[undersampled_idx]
        self.filename_array = self.filename_array[undersampled_idx]

            

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
        # self.dataset_dir = os.path.join("datasets/", self.dataset_name)
        self.dataset_dir = '/nobackup/sonic/celebA'
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


def get_celebA_dataloader(args, split):
    kwargs = {'pin_memory': True, 'num_workers': 8, 'drop_last': True}
    dataset = celebADataset(args, split)
    dataloader = DataLoader(dataset=dataset,
                                batch_size=args.batch_size,
                                shuffle=True,
                                **kwargs)
    return dataloader

    


def get_celebA_ood_dataloader(args):
    kwargs = {'pin_memory': True, 'num_workers': 8, 'drop_last': True}
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
    args = parser.parse_args()

    dataloader = get_celebA_dataloader(args, split='train')
    ood_dataloader = get_celebA_ood_dataloader(args)
    