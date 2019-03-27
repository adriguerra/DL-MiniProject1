from dlc_practical_prologue import *

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
