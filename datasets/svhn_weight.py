"""Dataset setting and data loader for SVHN."""

import torch
from torchvision import datasets, transforms
import os
class SVHN(datasets.SVHN):
    def __init__(self,*args,**kwargs):
        super().__init__(*args, **kwargs)
    def __getitem__(self, index):
        data, target = super().__getitem__(index)
        return data, target, index

def get_svhn_weight(params, train, weights):
    """Get SVHN datasets loader."""
    # image pre-processing
    dataset_root, batch_size = params.dataset_root, params.batch_size
    pre_process = transforms.Compose([transforms.Resize(32),
                                      transforms.ToTensor(),
                                      transforms.Normalize(
                                          mean=[0.4376, 0.4438, 0.4729],
                                          std=[0.1201, 0.1231, 0.1052]
                                      )])
    
    # datasets and data loader
    if train:
        svhn_dataset = SVHN(root=os.path.join(dataset_root),
                                   split='train',
                                   transform=pre_process,
                                   download=True)
    else:
        svhn_dataset = SVHN(root=os.path.join(dataset_root),
                                   split='test',
                                   transform=pre_process,
                                   download=True)
    num_sample = len(svhn_dataset)
    if len(weights) == 10: 
        sample_weight = torch.tensor([weights[label] for label in svhn_dataset.labels])
        svhn_data_loader = torch.utils.data.DataLoader(
            dataset=svhn_dataset,
            batch_size=batch_size,
            sampler=torch.utils.data.sampler.WeightedRandomSampler(
                sample_weight,len(sample_weight)),
            drop_last=True)
    else: 
        svhn_data_loader = torch.utils.data.DataLoader(
            dataset=svhn_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True)
    return svhn_data_loader, num_sample
