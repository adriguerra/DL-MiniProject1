import torch
import argparse
import os

from torchvision import datasets

mini_batch_size = 100

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

def preprocess_data(x_train, y_train, x_test, y_test, reshape, one_hot_encoded, split, normalized):
    """
    Preprocesses and formats all the data as appropriate for the architecture.

    IMPORTANT: Returns arrays of tensors.

    :param x_train: tensor N × 2 × 14 × 14
    :param y_train: tensor N × 2
    :param x_test: tensor N × 2 × 14 × 14
    :param y_test: tensor N × 2
    :param reshape: reshapes x_train and x_test to tensors of dimension N × 2 × 196
    :param one_hot_encoded: converts all tensors to one hot encoded tensors
    :param split: splits all tensors
    :param normalized: normalizes all tensors
    :returns: x_train, y_train, x_test, y_test
    Each variable is either an array of 2 tensors if split == True
    or an array of 1 tensor if split == False
    x_train, y_train, x_test, y_test
    """

    if reshape:
        x_train, x_test = reshape_data(x_train, x_test)

    if split:
        x_train, y_train, x_test, y_test = split_img_data(x_train, y_train, x_test, y_test)
    else:
        x_train = x_train.clone().reshape(x_train.size(0), -1)
        y_train = y_train.clone().reshape(y_train.size(0), -1)
        x_test = x_test.clone().reshape(x_test.size(0), -1)
        y_test = y_test.clone().reshape(y_test.size(0), -1)

    x_train[0] = 0.9*x_train[0]
    x_train[1] = 0.9*x_train[1]

    x_test[0] = 0.9*x_test[0]
    x_test[1] = 0.9*x_test[1]

    if one_hot_encoded:
        y_train[0] = convert_to_one_hot_labels(x_train[0], y_train[0])
        y_train[1] = convert_to_one_hot_labels(x_train[1], y_train[1])

        y_test[0] = convert_to_one_hot_labels(x_test[0], y_test[0])
        y_test[1] = convert_to_one_hot_labels(x_test[1], y_test[1])

    if normalized:
        x_train[0], x_test[0] = normalize(x_train[0], x_test[0])
        x_train[1], x_test[1] = normalize(x_train[1], x_test[1])

    return x_train, y_train, x_test, y_test


def convert_to_one_hot_labels(input, target):
    """

    :param input: the input of our data
    :param target: the target to be one hot encoded
    :return:
    """
    tmp = input.new_zeros(target.size(0), target.max() + 1)
    tmp.scatter_(1, target.view(-1, 1), 1.0)
    return tmp


def normalize(x_train, x_test):
    """

    :param x_train: The train input we want to normalize
    :param x_test: the test input we want to normalize
    :return: train input and test input normalized
    """
    mu, std = x_train.mean(), x_train.std()
    x_train.sub_(mu).div_(std)
    x_test.sub_(mu).div_(std)
    return x_train, x_test


def reshape_data(x_train, x_test):
    """

    :param x_train: input train we want to reshape
    :param x_test: test train we want to reshape
    :return:
    """
    x_train = x_train.clone().reshape(x_train.size(0), 2, -1)
    x_test = x_test.clone().reshape(x_test.size(0), 2, -1)
    return x_train, x_test


def split_img_data(x_train, y_train, x_test, y_test):
    """

    :param x_train: the input train we want to split into two images
    :param y_train: the target train we want to split into two images
    :param x_test: the input test we want to split
    :param y_test: the target test we want to split
    :return: splitted data
    """
    x_train1 = x_train[:, 0]
    x_train2 = x_train[:, 1]

    x_test1 = x_test[:, 0]
    x_test2 = x_test[:, 1]

    y_train1 = y_train[:,0]
    y_train2 = y_train[:,1]

    y_test1 = y_test[:,0]
    y_test2 = y_test[:,1]

    return [x_train1, x_train2], [y_train1, y_train2], [x_test1, x_test2], [y_test1, y_test2]


def compute_errors(m, x_train, y_train, x_test, y_test, stds, one_hot_encoded):
    """Computes the train errors and test errors of the given models
    for the given standard deviations on the x_train, x_test data
    with the labels in the y_train and y_test.
    If the standard deviations parameter is None, a Xavier
    initialization is performed."""

    model = m()
    if(stds is None):
        for p in model.parameters():
            if len(p.size()) == 2:
                updated_tensor, std = xavier_normal_(p.data.normal_(0, 1), gain=1)
        print("Computed standard deviation according to 'Xavier initialization': {:.3f}\n".format(std))
        stds = [std]
    for std in stds:
        if std > 0:
            for p in model.parameters():
                p.data.normal_(0, std)

        train_model(model, x_train, y_train, one_hot_encoded)
        train_error = compute_nb_errors(model, x_train, y_train, one_hot_encoded)
        test_error = compute_nb_errors(model, x_test, y_test, one_hot_encoded)
        print('std {:f} {:s} train_error {:.02f}% test_error {:.02f}%'.format(
                std,
                m.__name__,
                train_error / x_train.size(0) * 100,
                test_error / x_test.size(0) * 100))

        
def compare_and_predict(output1, output2):
    """Compares the entries of the two outputs and returns 0 if entry of output1 is larger 
    than output21 and 1 if entry of output2 is larger."""
    predict = []
    
    for (a,b) in zip(output1, output2):
        if a <= b:
            predict.append(1)
        else:
            predict.append(0)
    return predict
       
    
def compute_error_(predicted, test):
    """Computes percentage of number of errors."""
    error = 0
    
    for (a,b) in zip(predicted, test):
        if a != b:
            error+=1
    return 100*(error/len(predicted))


def prevent_vanishing_gradient(train_input, test_input, train_classes, test_classes):
    """

    :param train_input: training input as list of two tensor images
    :param test_input: testing input as list of two tensor images
    :param train_classes: training digit target as list of two tensors
    :param test_classes: testing digit target as list of two tensors
    :return: multiplying data by 0.9 to prevent vanishing gradient
    """

    train_input = [train_input[0]*0.9, train_input[1]*0.9]
    test_input = [test_input[0] * 0.9, test_input[1] * 0.9]
    train_classes = [train_classes[0] * 0.9, train_classes[1] * 0.9]
    test_classes = [test_classes[0] * 0.9, test_classes[1] * 0.9]

    return train_input, test_input, train_classes, test_classes


def preprocess_data(train_input, test_input, train_classes, test_classes):
    """

    :param train_input: training input as list of two tensor images
    :param test_input: testing input as list of two tensor images
    :param train_classes: training digit target as list of two tensors
    :param test_classes: testing digit target as list of two tensors
    :return: preprocessed data
    """

    x, y, z, t = split_img_data(train_input, test_input, train_classes, test_classes)
    prevent_vanishing_gradient(x, y, z, t)
    x[0], y[0] = normalize(x[0], y[0])
    x[1], y[1] = normalize(x[1], y[1])
    x[0] = torch.unsqueeze(x[0], 1)
    x[1] = torch.unsqueeze(x[1], 1)
    y[0] = torch.unsqueeze(y[0], 1)
    y[1] = torch.unsqueeze(y[1], 1)

    return x, y, z, t


def compute_nb_errors(prediction, target):
    """

    :param prediction: the prediction of our model
    :param target: the real target
    :return:
    """
    errors = 0
    for (a, b) in zip(prediction, target):
        if a.float() != b.float():
            errors += 1
    return errors / len(prediction) * 100






