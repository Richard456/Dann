
import torch
from torchvision import transforms
import torch.utils.data as data
from PIL import Image
import os
import numpy as np

def get_cifar10_c(dataset_root, batch_size, train, noise_type):
    """Get CIFAR10_C datasets loader."""
    
    # image pre-processing
    pre_process = transforms.Compose([transforms.Resize(28),
                                      transforms.ToTensor(),
                                      transforms.Normalize(
                                          mean=(0.491, 0.482, 0.446), 
                                          std=(0.202, 0.199, 0.201)
                                      )
                                      ])

    # datasets and data loader
    cifar10_c_dataset = GetLoader(
        data_root=os.path.join(dataset_root),
        noise_type=noise_type,
        transform=pre_process)
    
    print("dataset len: ",len(cifar10_c_dataset))

    train_size = int(0.8 * len(cifar10_c_dataset))
    test_size = len(cifar10_c_dataset) - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(cifar10_c_dataset, [train_size, test_size])
    print("train len: ",len(train_dataset))
    print("test len: ",len(test_dataset))
    if train:
        cifar10_c_dataset=train_dataset
    else:
        cifar10_c_dataset=test_dataset

    cifar10_c_data_loader = torch.utils.data.DataLoader(
        dataset=cifar10_c_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=8)

    return cifar10_c_data_loader


class GetLoader(data.Dataset):
    def __init__(self, data_root, noise_type,transform=None):
        self.root = data_root
        self.transform = transform

        self.datafile=os.path.join(self.root,"CIFAR-10-C", noise_type+ ".npy")
        self.labelfile=os.path.join(self.root,"CIFAR-10-C", "labels.npy")

        self.imgs=np.load(self.datafile)
        self.labels=np.load(self.labelfile)

    def __getitem__(self, item):
        imgs, labels= self.imgs[item], self.labels[item]
        imgs = Image.fromarray(imgs, 'RGB')
        
        if self.transform is not None:
            imgs = self.transform(imgs)
            labels = int(labels)

        return imgs, labels

    def __len__(self):
        return len(self.labels)

