import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from datasets import get_mnist, get_mnistm, get_svhn
from datasets.usps import get_usps
from datasets.mnist_weight import get_mnist_weight
from datasets.mnistm_weight import get_mnistm_weight
from datasets.svhn_weight import get_svhn_weight
from datasets.usps_weight import get_usps_weight

from datasets.office import get_office
from datasets.officecaltech import get_officecaltech
from datasets.syndigits import get_syndigits
from datasets.synsigns import get_synsigns
from datasets.gtsrb import get_gtsrb

def get_dataset_root(): 
    return '/nobackup/yguo/datasets'

# data mode: 
# 0. All one (uniform samplings)
# 1. First half 0, second half 1
# 2. First half 1, second half 0
# 3. Overlapping support: 0-7 --> 2-9
# 4. Mild random weight
# 5. Strong random weight


# run mode: 
# 0. DANN
# 1. source weighting + method 1 
# 2. target weighting + method 2 
# 3. winsorization + method 3


def get_model_root(model_name, data_mode, run_mode):
    data_mode = 'data{}'.format(data_mode)
    run_mode = 'run{}'.format(run_mode)
    model_root = os.path.expanduser(os.path.join('runs', model_name, data_mode, run_mode))
    return model_root

def get_data(mode): 
    # Return a tuple of lists, where the first list corresponds to the source 
    # weight, and the second part corresponds to the target weight
    if mode == 0: 
        source_weight = torch.ones(10) 
        target_weight = torch.ones(10)
        return (source_weight, target_weight)
    elif mode == 1: 
        source_weight = torch.ones(10)
        target_weight = torch.ones(10)
        return (source_weight, target_weight)
    elif mode == 2:
        source_weight = torch.ones(10)
        target_weight = torch.tensor([1,1,1,1,1,0,0,0,0,0])
        return (source_weight, target_weight)
    elif mode == 3:
        source_weight = torch.tensor([1,1,1,1,1,1,1,1,0,0])
        target_weight = torch.tensor([0,0,1,1,1,1,1,1,1,1])
        return (source_weight, target_weight)
    elif mode == 4: 
        value = 0.25 
        source_weight = torch.ones(10)
        target_weight = torch.tensor(np.concatenate(
            [[value], np.random.uniform(value, 1-value, 8), [1-value]]))
        return (source_weight, target_weight)
    elif mode == 5:
        value = 0.0625
        source_weight = torch.ones(10)
        target_weight = torch.tensor(np.concatenate(
            [[value], np.random.uniform(value, 1-value, 8), [1-value]]))
        return (source_weight, target_weight)
    else:
        source_weight = torch.tensor([0.1,0.1,0.1,0.1,0.1, 1,1,1,1,1])
        target_weight = torch.tensor([0,0,0,0,0,1,1,1,1,1])
        return (source_weight, target_weight)
        

def make_cuda(tensor):
    """Use CUDA if it's available."""
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return tensor


def denormalize(x, std, mean):
    """Invert normalization, and then convert array into image."""
    out = x * std + mean
    return out.clamp(0, 1)


def init_weights(layer):
    """Init weights for layers w.r.t. the original paper."""
    layer_name = layer.__class__.__name__
    if layer_name.find("Conv") != -1:
        layer.weight.data.normal_(0.0, 0.02)
        layer.bias.data.fill(0.0)
    elif layer_name.find("BatchNorm") != -1:
        layer.weight.data.normal_(1.0, 0.02)
        layer.bias.data.fill_(0)


def init_random_seed(seed):
    if seed == None:
        seed = random.randint(1, 10000)
    print("use random seed: {}".format(seed))
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    




def get_data_loader(name, dataset_root, batch_size, train=True):
    """Get data loader by name."""
    if name == "mnist":
        return get_mnist(dataset_root, batch_size, train)
    elif name == "mnistm":
        return get_mnistm(dataset_root, batch_size, train)
    elif name == "svhn":
        return get_svhn(dataset_root, batch_size, train)
    elif name == "amazon31":
        return get_office(dataset_root, batch_size, 'amazon')
    elif name == "webcam31":
        return get_office(dataset_root, batch_size, 'webcam')
    elif name == "webcam10":
        return get_officecaltech(dataset_root, batch_size, 'webcam')
    elif name == "syndigits":
        return get_syndigits(dataset_root, batch_size, train)
    elif name == "synsigns":
        return get_synsigns(dataset_root, batch_size, train)
    elif name == "gtsrb":
        return get_gtsrb(dataset_root, batch_size, train)
    elif name == "usps": 
        return get_usps(dataset_root, batch_size, train)


def get_data_loader_weight(name, dataset_root, batch_size, train=True, weights = torch.tensor([]), subsample_size = 0):
    """Get data loader by name. If len(weights) is 0 (default), no weighted 
    sampling is performed.
    """
    if name == "mnist":
        return get_mnist_weight(dataset_root, batch_size, train, subsample_size = subsample_size ,weights = weights)
    elif name == "mnistm":
        return get_mnistm_weight(dataset_root, batch_size, train, weights = weights)
    elif name == "svhn":
        return get_svhn_weight(dataset_root, batch_size, train, weights = weights)
    elif name == "usps": 
        return get_usps_weight(dataset_root, batch_size, train, subsample_size = subsample_size, weights = weights)
    elif name == "amazon31":
        return get_office(dataset_root, batch_size, 'amazon')
    elif name == "webcam31":
        return get_office(dataset_root, batch_size, 'webcam')
    elif name == "webcam10":
        return get_officecaltech(dataset_root, batch_size, 'webcam')
    elif name == "syndigits":
        return get_syndigits(dataset_root, batch_size, train)
    elif name == "synsigns":
        return get_synsigns(dataset_root, batch_size, train)
    elif name == "gtsrb":
        return get_gtsrb(dataset_root, batch_size, train)

def init_model(net, restore):
    """Init models with cuda and weights."""
    # init weights of model
    # net.apply(init_weights)

    # restore model weights
    if restore is not None and os.path.exists(restore):
        net.load_state_dict(torch.load(restore))
        net.restored = True
        print("Restore model from: {}".format(os.path.abspath(restore)))
    else:
        print("No trained model, train from scratch.")

    # check if cuda is available
    if torch.cuda.is_available():
        cudnn.benchmark = True
        net.cuda()

    return net


def save_model(net, model_root, filename):
    """Save trained model."""
    if not os.path.exists(model_root):
        os.makedirs(model_root)
    torch.save(net.state_dict(), os.path.join(model_root, filename))
    print("save pretrained model to: {}".format(os.path.join(model_root, filename)))
