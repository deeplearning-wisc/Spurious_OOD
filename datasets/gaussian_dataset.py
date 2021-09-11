from torch.utils.data.dataset import Dataset
from torchvision import transforms
import torch 

class GaussianDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_size, img_size = 32, labels = None, transform = None, num_classes = 2):
            if labels == None:
                self.labels = (torch.ones(dataset_size) * num_classes).long()
            else:
                self.labels = labels
            images = torch.normal(0.5, 0.5, size=(dataset_size,3,img_size,img_size))
            self.images = torch.clamp(images, 0, 1)
            self.transform = transform

    def __len__(self):
            return len(self.images)

    def __getitem__(self, index):
            # Load data and get label
            if self.transform:
                X = self.transform(self.images[index])
            else:
                X = self.images[index]
            y = self.labels[index]

            return X, y