from dlc_practical_prologue import *
import torch
import math

from torch import optim
from torch import Tensor
from torch import nn

mini_batch_size = 100

def normalize(train_input, test_input):
    mu, std = train_input.mean(), train_input.std()
    train_input.sub_(mu).div_(std)
    test_input.sub_(mu).div_(std)
    return train_input, test_input

def preprocess_data(train_input, train_classes, test_input, test_classes):
    train_input = train_input.clone().reshape(train_input.size(0), 2, -1)
    test_input = test_input.clone().reshape(test_input.size(0), 2, -1)

    train_input1 = train_input[:, 0]
    train_input2 = train_input[:, 1]

    test_input1 = test_input[:, 0]
    test_input2 = test_input[:, 1]

    train_classes1 = train_classes[:,0]
    train_classes2 = train_classes[:,1]

    test_classes1 = test_classes[:,0]
    test_classes2 = test_classes[:,1]

    train_input1 = 0.9*train_input1
    train_input2 = 0.9*train_input2

    test_input1 = 0.9*test_input1
    test_input2 = 0.9*test_input2

    train_classes1 = convert_to_one_hot_labels(train_input1, train_classes1)
    train_classes2 = convert_to_one_hot_labels(train_input2, train_classes2)

    test_classes1 = convert_to_one_hot_labels(test_input1, test_classes1)
    test_classes2 = convert_to_one_hot_labels(test_input2, test_classes2)

    train_input1, test_classes1 = normalize(train_input1, test_classes1)
    train_input2, test_classes2 = normalize(train_input2, test_classes2)

    return train_input1, train_input2, train_classes1, train_classes2, test_input1, test_input2, test_classes1, test_classes2

def xavier_normal_(tensor, gain):
    fan_in = tensor.size()[0]
    fan_out = tensor.size()[1]
    std = gain * math.sqrt(2.0/(fan_in + fan_out))
    with torch.no_grad():
        return tensor.normal_(0,std), std

def train_model(model, train_input, train_target):
    model.train() 
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-1)
    nb_epochs = 250

    for e in range(nb_epochs):
        for b in range(0, train_input.size(0), mini_batch_size):
            output = model(train_input.narrow(0, b, mini_batch_size))
            # max needed if train_target is one-hot encoded
            loss = criterion(output, train_target.narrow(0, b, mini_batch_size).max(1)[1])
            model.zero_grad()
            loss.backward()
            optimizer.step()

def compute_nb_errors(model, data_input, data_target):

    nb_data_errors = 0

    for b in range(0, data_input.size(0), mini_batch_size):
        output = model(data_input.narrow(0, b, mini_batch_size))
        _, predicted_classes = torch.max(output.data, 1)
        for k in range(mini_batch_size):
            # max needed if one-hot encoded
            if data_target.data[b + k].max(0)[1] != predicted_classes[k]:
                nb_data_errors = nb_data_errors + 1

    return nb_data_errors

def compute_errors(m, train_input, train_classes, test_input, test_classes, stds):
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

        train_model(model, train_input, train_classes)
        print('std {:f} {:s} train_error {:.02f}% test_error {:.02f}%'.format(
                std,
                m.__name__,
                compute_nb_errors(model, train_input,
                                  train_classes) / train_input.size(0) * 100,
                compute_nb_errors(model, test_input, test_classes) / test_input.size(0) * 100))
