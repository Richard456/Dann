"""Dataset setting and data loader for MNIST."""


import torch
from torchvision import datasets, transforms
import os

def get_mnist(dataset_root, batch_size, train):
    """Get MNIST datasets loader."""
    # image pre-processing
    pre_process = transforms.Compose([transforms.Resize(28), # different img size settings for mnist(28), usps(16) and svhn(32).
                                      transforms.ToTensor(),
                                      transforms.Normalize(
                                          mean=[0.1307], # Mean of MNIST train data
                                          std=[0.3015] # std of MNIST train data
                                      )])

    # datasets and data loader
    mnist_dataset = datasets.MNIST(root=os.path.join(dataset_root),
                                   train=train,
                                   transform=pre_process,
                                   download=True)


    mnist_data_loader = torch.utils.data.DataLoader(
        dataset=mnist_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=8)

    return mnist_data_loader