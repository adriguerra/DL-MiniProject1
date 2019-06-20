from helper_functions import *
from torch import nn
from torch.nn import functional as F

# Number of training samples
N = 1000
# Create training and testing datasets
train_input, train_target, train_classes, test_input, test_target, test_classes = generate_pair_sets(N)
train_input, test_input, train_classes, test_classes = preprocess_data(train_input,
                                                                       test_input,
                                                                       train_classes,
                                                                       test_classes)

class CNNWWS(nn.Module):
    """Convolutional Neural Network without weight sharing."""

    def __init__(self):
        super().__init__()
        nb_hidden = 100
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv1_bn = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 32, kernel_size=2)
        self.conv2_bn = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=2)
        self.conv3_bn = nn.BatchNorm2d(64)

        self.drop1 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(64, nb_hidden)
        self.drop2 = nn.Dropout(p=0.5)


    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1_bn(self.conv1(x)), kernel_size=2))
        x = F.relu(F.max_pool2d(self.conv2_bn(self.conv2(x)), kernel_size=2))
        x = F.relu(self.conv3_bn(self.conv3(x)))
        x = self.drop1(x)
        x = F.relu(self.fc1(x.view(-1, 64)))

        x = self.drop2(x)
        return x

def train_model(model1, model2, train_input1, train_input2, train_target1, train_target2, train_target3,
                mini_batch_size, classifier1, classifier2, classifier3, digit_scalar=1,
                binary_target_scalar=1):
    """
    Method to train the models of two images.

    :param model1: the model that trains the first image
    :param model2: the model that trains the second image
    :param train_input1: the training input of our first image
    :param train_input2: the training input of our second image
    :param train_target1: the training target of our first image
    :param train_target2: the training target of our second image
    :param train_target3: the training target of the oredering
    :param mini_batch_size: the batch size on which the sgd is trained
    :param classifier1: the classifier of the first digit
    :param classifier2: the classifier of the second digit
    :param classifier3: the classifier of the main ordering
    :param digit_scalar: the weight used to calibrate the loss of the digit prediction
    :param binary_target_scalar: the weight used to calibrate the loss of the ordering prediction
    """
    criterion = nn.CrossEntropyLoss()
    eta = 1e-1
    optimizer1 = torch.optim.SGD(model1.parameters(), lr=eta, momentum=0)
    optimizer2 = torch.optim.SGD(model2.parameters(), lr=eta, momentum=0) # check the lectures

    for e in range(25):
        sum_loss = 0
        for b in range(0, train_input1.size(0), mini_batch_size):
            encoded_img1 = model1(train_input1.narrow(0, b, mini_batch_size))
            encoded_img2 = model2(train_input2.narrow(0, b, mini_batch_size))

            output_x = classifier1(encoded_img1)
            output_y = classifier2(encoded_img2)
            input = torch.cat([encoded_img1, encoded_img2], 1)
            output_binary_target = classifier3(input)
            loss_x = criterion(output_x, train_target1.narrow(0, b, mini_batch_size).long())
            loss_y = criterion(output_y, train_target2.narrow(0, b, mini_batch_size).long())
            loss_binary_target = criterion(output_binary_target, train_target3.narrow(0, b, mini_batch_size).long())
            loss = digit_scalar * (loss_x + loss_y) + binary_target_scalar * loss_binary_target

            model1.zero_grad()
            model2.zero_grad()

            loss.backward()
            sum_loss = sum_loss + loss.item()
            optimizer1.step()
            optimizer2.step()


def train_without_ws(digit_scalar):
    """
    Train without weight-sharing.

    :param digit_scalar: the digit scalar calibrating the auxiliary loss, 0 if we don't want to use auxiliary loss
    """
    print("----Train model----")
    model1 = CNNWWS()
    model2 = CNNWWS()
    classifier1 = nn.Sequential(nn.Linear(100, 10), nn.Sigmoid())
    classifier2 = nn.Sequential(nn.Linear(100, 10), nn.Sigmoid())
    classifier3 = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(100 * 2, 2), nn.Sigmoid())

    print("Auxiliary loss = " + str(digit_scalar))

    train_model(model1, model2, train_input[0], train_input[1], train_classes[0], train_classes[1], train_target,
                mini_batch_size, classifier1, classifier2, classifier3, digit_scalar)
    model1.eval()
    model2.eval()
    classifier3.eval()
    encoder1 = model1(test_input[0])
    encoder2 = model2(test_input[1])
    output1 = classifier1(encoder1)
    output2 = classifier2(encoder2)
    prediction = classifier3(torch.cat([encoder1, encoder2], 1))

    print("Accuracy based on classes prediction:")
    print(100-compute_error_(compare_and_predict(output1.max(1)[1], output2.max(1)[1]), test_target))
    print("Accuracy based on target prediction:")
    print(100-compute_error_(prediction.max(1)[1], test_target))

    return compute_error_(compare_and_predict(output1.max(1)[1], output2.max(1)[1]), test_target), compute_error_(prediction.max(1)[1], test_target)
