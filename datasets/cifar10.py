"""Dataset setting and data loader for MNIST."""


import torch
from torchvision import datasets, transforms
import os

def get_cifar10(dataset_root, batch_size, train):
    """Get CIFAR datasets loader."""
    # image pre-processing
    pre_process = transforms.Compose([transforms.Resize(28),
                                      transforms.ToTensor(),
                                      transforms.Normalize(
                                          mean=(0.491, 0.482, 0.446), 
                                          std=(0.202, 0.199, 0.201)
                                      )
                                      ])

    # datasets and data loader
    cifar10_dataset = datasets.CIFAR10(root=os.path.join(dataset_root),
                                   train=train,
                                   transform=pre_process,
                                   download=True)


    cifar10_data_loader = torch.utils.data.DataLoader(
        dataset=cifar10_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=8)

    return cifar10_data_loader
