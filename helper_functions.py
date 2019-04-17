import torch
import math
import argparse
import os
import torch.nn.functional as F

from torchvision import datasets
from torch import optim
from torch import Tensor
from torch import nn


mini_batch_size = 100

# Methods from dlc_practical_prologue
######################################################################

parser = argparse.ArgumentParser(description='DLC prologue file for practical sessions.')

parser.add_argument('--full',
                    action='store_true', default=False,
                    help = 'Use the full set, can take ages (default False)')

parser.add_argument('--tiny',
                    action='store_true', default=False,
                    help = 'Use a very small set for quick checks (default False)')

parser.add_argument('--seed',
                    type = int, default = 0,
                    help = 'Random seed (default 0, < 0 is no seeding)')

parser.add_argument('--cifar',
                    action='store_true', default=False,
                    help = 'Use the CIFAR data-set and not MNIST (default False)')

parser.add_argument('--data_dir',
                    type = str, default = None,
                    help = 'Where are the PyTorch data located (default $PYTORCH_DATA_DIR or \'./data\')')

# Timur's fix
parser.add_argument('-f', '--file',
                    help = 'quick hack for jupyter')

args = parser.parse_args()

if args.seed >= 0:
    torch.manual_seed(args.seed)

######################################################################
# The data

def convert_to_one_hot_labels(input, target):
    tmp = input.new_zeros(target.size(0), target.max() + 1)
    tmp.scatter_(1, target.view(-1, 1), 1.0)
    return tmp

def load_data(cifar = None, one_hot_labels = False, normalize = False, flatten = True):

    if args.data_dir is not None:
        data_dir = args.data_dir
    else:
        data_dir = os.environ.get('PYTORCH_DATA_DIR')
        if data_dir is None:
            data_dir = './data'

    if args.cifar or (cifar is not None and cifar):
        print('* Using CIFAR')
        cifar_train_set = datasets.CIFAR10(data_dir + '/cifar10/', train = True, download = True)
        cifar_test_set = datasets.CIFAR10(data_dir + '/cifar10/', train = False, download = True)

        train_input = torch.from_numpy(cifar_train_set.train_data)
        train_input = train_input.transpose(3, 1).transpose(2, 3).float()
        train_target = torch.tensor(cifar_train_set.train_labels, dtype = torch.int64)

        test_input = torch.from_numpy(cifar_test_set.test_data).float()
        test_input = test_input.transpose(3, 1).transpose(2, 3).float()
        test_target = torch.tensor(cifar_test_set.test_labels, dtype = torch.int64)

    else:
        print('* Using MNIST')
        mnist_train_set = datasets.MNIST(data_dir + '/mnist/', train = True, download = True)
        mnist_test_set = datasets.MNIST(data_dir + '/mnist/', train = False, download = True)

        train_input = mnist_train_set.train_data.view(-1, 1, 28, 28).float()
        train_target = mnist_train_set.train_labels
        test_input = mnist_test_set.test_data.view(-1, 1, 28, 28).float()
        test_target = mnist_test_set.test_labels

    if flatten:
        train_input = train_input.clone().reshape(train_input.size(0), -1)
        test_input = test_input.clone().reshape(test_input.size(0), -1)

    if args.full:
        if args.tiny:
            raise ValueError('Cannot have both --full and --tiny')
    else:
        if args.tiny:
            print('** Reduce the data-set to the tiny setup')
            train_input = train_input.narrow(0, 0, 500)
            train_target = train_target.narrow(0, 0, 500)
            test_input = test_input.narrow(0, 0, 100)
            test_target = test_target.narrow(0, 0, 100)
        else:
            print('** Reduce the data-set (use --full for the full thing)')
            train_input = train_input.narrow(0, 0, 1000)
            train_target = train_target.narrow(0, 0, 1000)
            test_input = test_input.narrow(0, 0, 1000)
            test_target = test_target.narrow(0, 0, 1000)

    print('** Use {:d} train and {:d} test samples'.format(train_input.size(0), test_input.size(0)))

    if one_hot_labels:
        train_target = convert_to_one_hot_labels(train_input, train_target)
        test_target = convert_to_one_hot_labels(test_input, test_target)

    if normalize:
        mu, std = train_input.mean(), train_input.std()
        train_input.sub_(mu).div_(std)
        test_input.sub_(mu).div_(std)

    return train_input, train_target, test_input, test_target

######################################################################

def mnist_to_pairs(nb, input, target):
    input = torch.functional.F.avg_pool2d(input, kernel_size = 2)
    a = torch.randperm(input.size(0))
    a = a[:2 * nb].view(nb, 2)
    input = torch.cat((input[a[:, 0]], input[a[:, 1]]), 1)
    classes = target[a]
    target = (classes[:, 0] <= classes[:, 1]).long()
    return input, target, classes

######################################################################

def generate_pair_sets(nb):
    if args.data_dir is not None:
        data_dir = args.data_dir
    else:
        data_dir = os.environ.get('PYTORCH_DATA_DIR')
        if data_dir is None:
            data_dir = './data'

    train_set = datasets.MNIST(data_dir + '/mnist/', train = True, download = True)
    train_input = train_set.train_data.view(-1, 1, 28, 28).float()
    train_target = train_set.train_labels

    test_set = datasets.MNIST(data_dir + '/mnist/', train = False, download = True)
    test_input = test_set.test_data.view(-1, 1, 28, 28).float()
    test_target = test_set.test_labels

    return mnist_to_pairs(nb, train_input, train_target) + \
           mnist_to_pairs(nb, test_input, test_target)

######################################################################

# Our methods
# TODO merge both
# TODO document methods

def normalize(train_input, test_input):
    mu, std = train_input.mean(), train_input.std()
    train_input.sub_(mu).div_(std)
    test_input.sub_(mu).div_(std)
    return train_input, test_input

def reshape_data(train_input, test_input):
    train_input = train_input.clone().reshape(train_input.size(0), 2, -1)
    test_input = test_input.clone().reshape(test_input.size(0), 2, -1)
    return train_input, test_input

def split_img_data(train_input, test_input, train_classes, test_classes):
    train_input1 = train_input[:, 0]
    train_input2 = train_input[:, 1]

    test_input1 = test_input[:, 0]
    test_input2 = test_input[:, 1]

    train_classes1 = train_classes[:,0]
    train_classes2 = train_classes[:,1]

    test_classes1 = test_classes[:,0]
    test_classes2 = test_classes[:,1]

    return train_input1, train_input2, test_input1, test_input2, train_classes1, train_classes2, test_classes1, test_classes2

def xavier_normal_(tensor, gain):
    fan_in = tensor.size()[0]
    fan_out = tensor.size()[1]
    std = gain * math.sqrt(2.0/(fan_in + fan_out))
    with torch.no_grad():
        return tensor.normal_(0,std), std

def train_model(model, train_input, train_target, one_hot_encoded):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-1)
    nb_epochs = 250

    for e in range(nb_epochs):
        for b in range(0, train_input.size(0), mini_batch_size):
            output = model(train_input.narrow(0, b, mini_batch_size))
            # max needed if train_target is one-hot encoded
            if(one_hot_encoded):
                loss = criterion(output, train_target.narrow(0, b, mini_batch_size).max(1)[1])
            else:
                loss = criterion(output, train_target.narrow(0, b, mini_batch_size))
            model.zero_grad()
            loss.backward()
            optimizer.step()

def compute_nb_errors(model, data_input, data_target, one_hot_encoded):

    nb_data_errors = 0

    for b in range(0, data_input.size(0), mini_batch_size):
        output = model(data_input.narrow(0, b, mini_batch_size))
        _, predicted_classes = torch.max(output.data, 1)
        for k in range(mini_batch_size):
            # max needed if one-hot encoded
            target = data_target.data[b + k].max(0)[1] if one_hot_encoded else data_target.data[b + k]
            if target != predicted_classes[k]:
                nb_data_errors = nb_data_errors + 1

    return nb_data_errors

def compute_errors(m, train_input, train_classes, test_input, test_classes, stds, one_hot_encoded):
    """Computes the train errors and test errors of the given models
    for the given standard deviations on the train_input, test_input data
    with the labels in the train_classes and test_classes.
    If the standard deviations parameter is None, a Xavier
    initialization is performed."""

    model = m()
    if(stds is None):
        for p in model.parameters():
            if len(p.size()) == 2:
                updated_tensor, std = xavier_normal_(p.data.normal_(0, 1), gain=1)
        print("Computed standard deviation according to 'Xavier initialization': {:.3f}".format(std))
        stds = [std]
    for std in stds:
        if std > 0:
            for p in model.parameters():
                p.data.normal_(0, std)

        train_model(model, train_input, train_classes, one_hot_encoded)
        train_error = compute_nb_errors(model, train_input, train_classes, one_hot_encoded)
        test_error = compute_nb_errors(model, test_input, test_classes, one_hot_encoded)
        print('std {:f} {:s} train_error {:.02f}% test_error {:.02f}%'.format(
                std,
                m.__name__,
                train_error / train_input.size(0) * 100,
                test_error / test_input.size(0) * 100))
