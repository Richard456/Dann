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


def get_data_loader_weight(name, dataset_root, batch_size, train=True, weights = torch.tensor([])):
    """Get data loader by name."""
    if name == "mnist":
        return get_mnist_weight(dataset_root, batch_size, train, weights = weights)
    elif name == "mnistm":
        return get_mnistm_weight(dataset_root, batch_size, train, weights = weights)
    elif name == "svhn":
        return get_svhn_weight(dataset_root, batch_size, train, weights = weights)
    elif name == "usps": 
        return get_usps_weight(dataset_root, batch_size, train, weights = weights)
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
