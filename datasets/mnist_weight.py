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
    
def get_mnist_weight(dataset_root, batch_size, train, subsample_size, weights):
    """Get MNIST datasets loader."""
    # image pre-processing
    pre_process = transforms.Compose([transforms.Resize(28), # different img size settings for mnist(28) and svhn(32).
                                      transforms.ToTensor(),
                                      transforms.Normalize(
                                      mean=[0.1307], # Mean of MNIST train data
                                      std=[0.3015] # std of MNIST train data
                                      )])

    # datasets and data loader
    mnist_dataset = MNIST(root=os.path.join(dataset_root),
                                   train=train,
                                   transform=pre_process,
                                   download=True)
    print("loading source train data")
    print(subsample_size)
    num_sample = len(mnist_dataset)
    if len(weights) ==10: 
        sample_weight = torch.tensor([weights[label] for label in mnist_dataset.targets])
        subsize = len(sample_weight) 
        if subsample_size != 0: 
            subsize = subsample_size
        print("MNIST")
        print("subsample size:{}".format(subsample_size))
        print("subsize {}".format(len(sample_weight)))
        mnist_data_loader = torch.utils.data.DataLoader(
            dataset=mnist_dataset,
            batch_size=batch_size,
            sampler=torch.utils.data.sampler.WeightedRandomSampler(
                sample_weight, subsize),
            drop_last=True,
            num_workers=8)
    else: 
        mnist_data_loader = torch.utils.data.DataLoader(
            dataset=mnist_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=8)
    return mnist_data_loader, num_sample
