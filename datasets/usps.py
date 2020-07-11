"""Dataset setting and data loader for MNIST."""


import torch
from torchvision import datasets, transforms
import os

def get_usps(dataset_root, batch_size, train):
    """Get USPS datasets loader."""
    # image pre-processing
    pre_process = transforms.Compose([transforms.Resize(28),
                                      transforms.ToTensor(),
                                      transforms.Normalize(
                                          mean=[0.2473], # Mean for USPS train data
                                          std=[0.2665] # std for USPS train data
                                      )
                                      ])

    # datasets and data loader
    usps_dataset = datasets.USPS(root=os.path.join(dataset_root),
                                   train=train,
                                   transform=pre_process,
                                   download=True)


    usps_data_loader = torch.utils.data.DataLoader(
        dataset=usps_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=8)

    return usps_data_loader
