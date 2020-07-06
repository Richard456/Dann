"""Dataset setting and data loader for MNIST."""


import torch
from torchvision import datasets, transforms
import os


class MNIST(datasets.MNIST):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        data, target = super().__getitem__(index)
        return data, target, index
    
def get_mnist_weight(dataset_root, batch_size, train, sampler = 'None'):
    """Get MNIST datasets loader."""
    # image pre-processing
    pre_process = transforms.Compose([transforms.Resize(32), # different img size settings for mnist(28) and svhn(32).
                                      transforms.ToTensor(),
                                      transforms.Normalize(
                                          mean=[0.5],
                                          std=[0.5]
                                      )])

    # datasets and data loader
    mnist_dataset = MNIST(root=os.path.join(dataset_root),
                                   train=train,
                                   transform=pre_process,
                                   download=True)
    num_sample = len(mnist_dataset)
    if sampler is not None:
        sampler.num_samples = num_sample



    mnist_data_loader = torch.utils.data.DataLoader(
        dataset=mnist_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        drop_last=True,
        num_workers=8)

    return mnist_data_loader, num_sample
