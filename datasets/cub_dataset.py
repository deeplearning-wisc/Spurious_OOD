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

class WaterbirdDataset(Dataset):
    def __init__(self, data_correlation, train=True):
        self.split_dict = {
            'train': 0,
            'val': 1,
            'test': 2
        }
        self.train = train
        self.dataset_name = "waterbird_complete"+"{:0.2f}".format(data_correlation)[-2:]+"_forest2water2"
        self.dataset_dir = os.path.join("datasets", self.dataset_name)
        if not os.path.exists(self.dataset_dir):
            raise ValueError(
                f'{self.dataset_dir} does not exist yet. Please generate the dataset first.') 
        self.metadata_df = pd.read_csv(
            os.path.join(self.dataset_dir, 'metadata.csv'))
        if self.train:
            self.metadata_df = self.metadata_df[self.metadata_df['split']==self.split_dict['train']]
        else:
            self.metadata_df = self.metadata_df[self.metadata_df['split']==self.split_dict['test']]
        
        self.y_array = self.metadata_df['y'].values
        self.filename_array = self.metadata_df['img_filename'].values
        self.transform = get_transform_cub(self.train)

    def __len__(self):
        return len(self.filename_array)
    
    def __getitem__(self, idx):
        y = self.y_array[idx]
        img_filename = os.path.join(
            self.dataset_dir,
            self.filename_array[idx])
        img = Image.open(img_filename).convert('RGB')
        img = self.transform(img)

        return img, y, y
    
def get_transform_cub(train):
    scale = 256.0/224.0
    target_resolution = (224, 224)
    assert target_resolution is not None

    if not train:
        # Resizes the image to a slightly larger square then crops the center.
        transform = transforms.Compose([
            transforms.Resize((int(target_resolution[0]*scale), int(target_resolution[1]*scale))),
            transforms.CenterCrop(target_resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(
                target_resolution,
                scale=(0.7, 1.0),
                ratio=(0.75, 1.3333333333333333),
                interpolation=2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    return transform

def get_waterbird_dataloader(args, data_label_correlation, train=True):
    kwargs = {'pin_memory': False, 'num_workers': 8, 'drop_last': True}
    dataset = WaterbirdDataset(data_correlation=data_label_correlation, train=train)
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

if __name__ == "__main__":
    print(len(WaterbirdDataset(data_correlation=0.95, train=True)))
    print(len(WaterbirdDataset(data_correlation=0.95, train=False)))