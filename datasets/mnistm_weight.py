"""Dataset setting and data loader for MNIST_M."""

import torch
from torchvision import transforms
import torch.utils.data as data
from PIL import Image
import os

class GetLoader(data.Dataset):
    def __init__(self, data_root, data_list, transform=None):
        self.root = data_root
        self.transform = transform

        f = open(data_list, 'r')
        data_list = f.readlines()
        f.close()

        self.n_data = len(data_list)

        self.img_paths = []
        self.img_labels = []

        for data in data_list:
            self.img_paths.append(data[:-3])
            self.img_labels.append(data[-2])

    def __getitem__(self, item):
        img_paths, labels = self.img_paths[item], self.img_labels[item]
        imgs = Image.open(os.path.join(self.root, img_paths)).convert('RGB')

        if self.transform is not None:
            imgs = self.transform(imgs)
            labels = int(labels)

        return imgs, labels, item

    def __len__(self):
        return self.n_data

def get_mnistm_weight(dataset_root, batch_size, train, sampler = None):
    """Get MNISTM datasets loader."""
    # image pre-processing
    pre_process = transforms.Compose([transforms.Resize(28),
                                     transforms.ToTensor(),
                                     transforms.Normalize(
                                          mean=(0.5, 0.5, 0.5),
                                          std=(0.5, 0.5, 0.5)
                                     )])
 
    # datasets and data_loader
    if train:
        train_list = os.path.join(dataset_root, 'mnist_m','mnist_m_train_labels.txt')
        mnistm_dataset = GetLoader(
            data_root=os.path.join(dataset_root, 'mnist_m', 'mnist_m_train'),
            data_list=train_list,
            transform=pre_process)
    else:
        train_list = os.path.join(dataset_root, 'mnist_m', 'mnist_m_test_labels.txt')
        mnistm_dataset = GetLoader(
            data_root=os.path.join(dataset_root, 'mnist_m', 'mnist_m_test'),
            data_list=train_list,
            transform=pre_process)
    num_sample = len(mnistm_dataset)
    if sampler is not None: 
        sampler.num_samples = num_sample
    mnistm_dataloader = torch.utils.data.DataLoader(
        dataset=mnistm_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=8)

    return mnistm_dataloader, num_sample
