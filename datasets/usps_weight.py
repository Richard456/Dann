"""Dataset setting and data loader for USPS."""


import torch
from torchvision import datasets, transforms
import os

import torch
from torchvision import datasets, transforms
import os

class USPS(datasets.USPS):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        data, target = super().__getitem__(index)
        return data, target, index
    
def get_usps_weight(dataset_root, batch_size, train, subsample_size, weights):
    """Get USPS datasets loader."""
    # image pre-processing
    pre_process = transforms.Compose([transforms.Resize(28),
                                      transforms.ToTensor(),
                                      transforms.Normalize(
                                      mean=[0.2473], # Mean for USPS train data
                                      std=[0.2665] # std for USPS train data
                                      )])

    # datasets and data loader
    usps_dataset = USPS(root=os.path.join(dataset_root),
                                   train=train,
                                   transform=pre_process,
                                   download=True)
    num_sample = len(usps_dataset)
 
    if len(weights) ==10: 
        sample_weight = torch.tensor([weights[label] for label in usps_dataset.targets])
        subsize = len(sample_weight)        
        if subsample_size != 0: 
            subsize = subsample_size
        print('usps')
        print("subsample size:{}".format(subsample_size))
        print("subsize {}".format(len(sample_weight)))
        usps_data_loader = torch.utils.data.DataLoader(
            dataset=usps_dataset,
            batch_size=batch_size,
            sampler=torch.utils.data.sampler.WeightedRandomSampler(
                sample_weight, subsize),
            drop_last=True,
            num_workers=8)
    else: 
        usps_data_loader = torch.utils.data.DataLoader(
            dataset=usps_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=8)
    return usps_data_loader, num_sample
