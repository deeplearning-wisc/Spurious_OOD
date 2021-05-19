"""ReBias
Copyright (c) 2020-present NAVER Corp.
MIT license

Python implementation of Biased-MNIST.
"""
import os
import numpy as np
from PIL import Image

import torch
from torch.utils import data

from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data.dataloader import default_collate
from torch.utils.data.distributed import DistributedSampler


# class PartialMNIST(MNIST):
#         def __init__(self, root, train = True, transform = None, target_transform=None, download = False, partial=False):
#             super().__init__(root, train=train, transform=transform,
#                             target_transform=target_transform,
#                             download=download)

class BiasedMNIST(MNIST):
    """A base class for Biased-MNIST.
    We manually select ten colours to synthetic colour bias. (See `COLOUR_MAP` for the colour configuration)
    Usage is exactly same as torchvision MNIST dataset class.

    You have two paramters to control the level of bias.

    Parameters
    ----------
    root : str
        path to MNIST dataset.
    data_label_correlation : float, default=1.0
        Here, each class has the pre-defined colour (bias).
        data_label_correlation, or `rho` controls the level of the dataset bias.

        A sample is coloured with
            - the pre-defined colour with probability `rho`,
            - coloured with one of the other colours with probability `1 - rho`.
              The number of ``other colours'' is controlled by `n_confusing_labels` (default: 9).
        Note that the colour is injected into the background of the image (see `_binary_to_colour`).

        Hence, we have
            - Perfectly biased dataset with rho=1.0
            - Perfectly unbiased with rho=0.1 (1/10) ==> our ``unbiased'' setting in the test time.
        In the paper, we explore the high correlations but with small hints, e.g., rho=0.999.

    n_confusing_labels : int, default=9
        In the real-world cases, biases are not equally distributed, but highly unbalanced.
        We mimic the unbalanced biases by changing the number of confusing colours for each class.
        In the paper, we use n_confusing_labels=9, i.e., during training, the model can observe
        all colours for each class. However, you can make the problem harder by setting smaller n_confusing_labels, e.g., 2.
        We suggest to researchers considering this benchmark for future researches.
    """

    # ORIGINAL_MAP = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [225, 225, 0], [225, 0, 225],
    #               [0, 255, 255], [255, 128, 0], [255, 0, 128], [128, 0, 255], [128, 128, 128]]
    # SET1: ONLY DIFF IN 0
    # COLOUR_MAP1 = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [225, 225, 0], [225, 0, 225],
    #               [255, 0, 0], [255, 0, 0],[255, 0, 0], [255, 0, 0], [255, 0, 0]]
    # COLOUR_MAP2 = [[128, 0, 255], [0, 255, 0], [0, 0, 255], [225, 225, 0], [225, 0, 225],
    #               [255, 0, 0], [255, 0, 0],[255, 0, 0], [255, 0, 0], [255, 0, 0]]
    # COLOUR_MAP2 = [[0, 0, 0], [255, 0, 0], [0, 0, 255], [225, 225, 0], [225, 0, 225],
    #               [255, 0, 0], [255, 0, 0],[255, 0, 0], [255, 0, 0], [255, 0, 0]]
    # SET2: ONLY DIFF IN 0 & 1
    COLOUR_MAP1 = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [225, 225, 0], [225, 0, 225],
                  [255, 0, 0], [255, 0, 0],[255, 0, 0], [255, 0, 0], [255, 0, 0]]
    COLOUR_MAP2 = [[128, 0, 255], [255, 0, 128], [0, 0, 255], [225, 225, 0], [225, 0, 225],
                  [255, 0, 0], [255, 0, 0],[0, 255, 0], [0, 255, 0], [0, 255, 0]] 
    COLOUR_MAP3 = [[0, 0, 255], [128, 0, 128], [0, 0, 255], [225, 225, 0], [225, 0, 225],
                  [255, 0, 0], [255, 0, 0],[0, 255, 0], [0, 255, 0], [0, 255, 0]]  
    COLOUR_MAP4 = [[255, 255, 0], [255, 192, 203], [0, 0, 255], [225, 225, 0], [225, 0, 225],
                  [255, 0, 0], [255, 0, 0],[0, 255, 0], [0, 255, 0], [0, 255, 0]]   
    # COLOUR_MAP2 = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], 
    #                 [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]


    def __init__(self, root, cmap, train=True, transform=None, target_transform=None,
                 download=False, data_label_correlation=1.0, n_confusing_labels=9, partial=False):
        super().__init__(root, train=train, transform=transform,
                         target_transform=target_transform,
                         download=download)
        self.cmap = cmap
        self.random = True
        # self.Train = train
        self.Partial = partial
        self.data_label_correlation = data_label_correlation
        self.n_confusing_labels = n_confusing_labels
        self.data, self.targets, self.biased_targets = self.build_biased_mnist()

        indices = np.arange(len(self.data))
        self._shuffle(indices)

        self.data = self.data[indices].numpy()
        self.targets = self.targets[indices]
        self.biased_targets = self.biased_targets[indices]

    @property
    def raw_folder(self):
        return os.path.join(self.root, 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, 'processed')

    def _shuffle(self, iteratable):
        if self.random:
            np.random.shuffle(iteratable)

    def _make_biased_mnist(self, indices, label, cmap):
        raise NotImplementedError

    def _update_bias_indices(self, bias_indices, label):
        if self.n_confusing_labels > 9 or self.n_confusing_labels < 1:
            raise ValueError(self.n_confusing_labels)

        indices = np.where((self.targets == label).numpy())[0] # find all the indices in self.targets (original MNIST) with the label (e.g. 0)
        self._shuffle(indices)
        indices = torch.LongTensor(indices) # still a list of indices, but with dtype = torch.int64

        n_samples = len(indices)    
        n_correlated_samples = int(n_samples * self.data_label_correlation)
        n_decorrelated_per_class = int(np.ceil((n_samples - n_correlated_samples) / (self.n_confusing_labels)))
        # print(f"######checking########### label: {label} total num of samples: {n_samples}; n_correlated_samples: {n_correlated_samples }; \
        #                                         n_decorrelated_per_class: {n_decorrelated_per_class} ")
        correlated_indices = indices[:n_correlated_samples]
        bias_indices[label] = torch.cat([bias_indices[label], correlated_indices]) # update bias_indices of the label. good trick: can cat empty tensor with a tensor list 

        decorrelated_indices = torch.split(indices[n_correlated_samples:], n_decorrelated_per_class)
        if self.Partial: # temp GDRO 
            other_labels = [_label % 2 for _label in range(label + 1, label + 1 + self.n_confusing_labels)]
        else:
            other_labels = [_label % 10 for _label in range(label + 1, label + 1 + self.n_confusing_labels)]
        self._shuffle(other_labels) # e.g. if current label = 0, then shuffled other_labels = [8, 4, 2, 7, 5, 1, 9, 6, 3]
                                    # if current label = 2, n_confusing_labels = 4, then other labels = [3,4,5,6]

        for idx, _indices in enumerate(decorrelated_indices): # ([5192. 4961, ...]. [...]. [...], [...]), each block stores indices that originally belongs to label
            _label = other_labels[idx] # we want to change the label of each block to one of other_labels
            bias_indices[_label] = torch.cat([bias_indices[_label], _indices])

    # core function in init()
    def build_biased_mnist(self):
        """Build biased MNIST.
        """
        if self.Partial: # temp GDRO
            n_labels = 2
        else: 
            n_labels = self.targets.max().item() + 1

        bias_indices = {label: torch.LongTensor() for label in range(n_labels)} #value of each key is an empty int64 tensor 
        for label in range(n_labels):
            self._update_bias_indices(bias_indices, label) #update the indices list corresponding to each label

        data = torch.ByteTensor()
        targets = torch.LongTensor()
        biased_targets = []

        for bias_label, indices in bias_indices.items():
            _data, _targets = self._make_biased_mnist(indices, bias_label, self.cmap) # now we have colorized images with target labels
            # if self.Partial: 
            #     if torch.any(_targets >= 5): # only works for correlation = 1
            #         continue
            data = torch.cat([data, _data])
            targets = torch.cat([targets, _targets]) # targets are the real target labels in MNIST
            biased_targets.extend([bias_label] * len(indices))

        biased_targets = torch.LongTensor(biased_targets)
        return data, targets, biased_targets

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img.astype(np.uint8), mode='RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, int(self.biased_targets[index])


class ColourBiasedMNIST(BiasedMNIST):
    def __init__(self, root, cmap, train=True, transform=None, target_transform=None,
                 download=False, data_label_correlation=1.0, n_confusing_labels=9, partial = False):
        super(ColourBiasedMNIST, self).__init__(root, train=train, transform=transform,
                                                target_transform=target_transform,
                                                download=download,
                                                data_label_correlation=data_label_correlation,
                                                n_confusing_labels=n_confusing_labels,
                                                partial = partial,
                                                cmap = cmap)

    def _binary_to_colour(self, data, colour):
        fg_data = torch.zeros_like(data) # e.g. shape of data with label 0: torch.Size([5923, 28, 28])
        fg_data[data != 0] = 255 # data != 0 is a mask which indicates the pixels that are not background; maximize the intensity of every point
        fg_data[data == 0] = 0  #background remains 0
        fg_data = torch.stack([fg_data, fg_data, fg_data], dim=1) # 1D to 3D

        bg_data = torch.zeros_like(data)
        bg_data[data == 0] = 1
        bg_data[data != 0] = 0
        bg_data = torch.stack([bg_data, bg_data, bg_data], dim=3) #bg_data is a 3D binary mask that indicates the pixels of the background area
        bg_data = bg_data * torch.ByteTensor(colour) # torch.ByteTensor(colour) e.g. tensor([255, 0, 0], dtype=torch.uint8); dtype of ByteTensor is unit8 i.e. 8-bit integer (unsigned)
        bg_data = bg_data.permute(0, 3, 1, 2)

        data = fg_data + bg_data
        return data.permute(0, 2, 3, 1)

    def _make_biased_mnist(self, indices, label, cmap):
        if cmap == "1": 
            label = self.COLOUR_MAP1[label]
            # indices = indices[: len(indices)//2]
        elif cmap == "2":
            label = self.COLOUR_MAP2[label]
            # indices = indices[len(indices)//2:]
        elif cmap == "3":
            label = self.COLOUR_MAP3[label]
        elif cmap == "4":
            label = self.COLOUR_MAP4[label]

        return self._binary_to_colour(self.data[indices], label), self.targets[indices]


def get_biased_mnist_dataloader(args, root, batch_size, data_label_correlation, cmap,
                                n_confusing_labels=9, train=True, partial=False):
    kwargs = {'pin_memory': False, 'num_workers': 8, 'drop_last': True}
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5),
                             std=(0.5, 0.5, 0.5))])
    dataset = ColourBiasedMNIST(root, train=train, transform=transform,
                                download=True, data_label_correlation=data_label_correlation,
                                n_confusing_labels=n_confusing_labels, partial=partial, cmap = cmap)
    if args.multi_gpu:
            ddp_sampler = DistributedSampler(dataset, num_replicas=args.n_gpus, rank=args.local_rank)
            dataloader = data.DataLoader(dataset=dataset,
                                    batch_size=batch_size,
                                    sampler=ddp_sampler,
                                    shuffle=False,
                                    **kwargs)
    else:
        dataloader = data.DataLoader(dataset=dataset,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    **kwargs)
    return dataloader

def generate_custom_ood_dataset(name, save_labels, data_label_correlation = 1, n_confusing_labels = 9, train=False, partial=False):
    loader = get_biased_mnist_dataloader(root = './datasets/MNIST', batch_size=1,
                                            data_label_correlation= data_label_correlation,
                                            n_confusing_labels= n_confusing_labels,
                                            train=train, partial=partial, cmap = "2")
    idx_to_class = {v: k for k, v in loader.dataset.class_to_idx.items()}
    result_path = f"datasets/ood_datasets/{name}"
    for i, (images, labels, _) in enumerate(loader):
        if labels not in save_labels:
            continue
        unorm = UnNormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        unorm(images[0])
        image_PIL = transforms.ToPILImage()(images[0])

        class_path = os.path.join(result_path, idx_to_class[labels.item()])
        if not os.path.exists(class_path):
            os.makedirs(class_path)
        image_PIL.save(os.path.join(class_path, f'img{i+1}.png'))


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

# example usage:

# tr_loader = get_biased_mnist_dataloader(root, batch_size=batch_size,
#                                             data_label_correlation=train_correlation,
#                                             n_confusing_labels=n_confusing_labels,
#                                             train=True)
# logger.log('preparing val loader...')
# val_loaders = {}
# val_loaders['biased'] = get_biased_mnist_dataloader(root, batch_size=batch_size,
#                                                     data_label_correlation=1,
#                                                     n_confusing_labels=n_confusing_labels,
#                                                     train=False)
# val_loaders['rho0'] = get_biased_mnist_dataloader(root, batch_size=batch_size,
#                                                   data_label_correlation=0,
#                                                   n_confusing_labels=9,
#                                                   train=False)
# val_loaders['unbiased'] = get_biased_mnist_dataloader(root, batch_size=batch_size,
#                                                       data_label_correlation=0.1,
#                                                       n_confusing_labels=9,
#                                                       train=False)
# logger.log('preparing trainer...')

if __name__ == "__main__":
    # batch_size = 20
    # train_correlation = 1
    # n_confusing_labels = 9
    # root = './datasets/MNIST'
    # tr_loader = get_biased_mnist_dataloader(root, batch_size=batch_size,
    #                                         data_label_correlation=train_correlation,
    #                                         n_confusing_labels=n_confusing_labels,
    #                                         train=True, partial=True)
    # generate_custom_ood_dataset("exam_train_set", save_labels = [0,1,2,3,4,5,6,7,8,9], data_label_correlation= 0.1,
    #                                         n_confusing_labels= 4, train=True, partial=True)
    generate_custom_ood_dataset("black0&1", save_labels=[0,1], data_label_correlation=1, n_confusing_labels= 4, train=True, partial=True)
