# Important imports
from helper_functions import *
from torch import nn
from torch.nn import functional as F

# the model of our network using weight sharing
class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
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
        self.fc2 = nn.Linear(nb_hidden, 10)
        self.drop3 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(nb_hidden * 2, 2)
        self.fc4 = nn.Sigmoid()

    def forward(self, x, y):
        x = F.relu(F.max_pool2d(self.conv1_bn(self.conv1(x)), kernel_size=2))
        y = F.relu(F.max_pool2d(self.conv1_bn(self.conv1(y)), kernel_size=2))

        x = F.relu(F.max_pool2d(self.conv2_bn(self.conv2(x)), kernel_size=2))
        y = F.relu(F.max_pool2d(self.conv2_bn(self.conv2(y)), kernel_size=2))

        x = F.relu(self.conv3_bn(self.conv3(x)))
        y = F.relu(self.conv3_bn(self.conv3(y)))

        x = self.drop1(x)
        y = self.drop1(y)

        x = F.relu(self.fc1(x.view(-1, 64)))
        y = F.relu(self.fc1(y.view(-1, 64)))

        x = self.drop2(x)
        y = self.drop2(y)

        binary_target = torch.cat([x, y], 1)

        x = self.fc2(x)
        y = self.fc2(y)

        #
        x = self.fc4(x)
        y = self.fc4(y)
        binary_target = self.drop3(binary_target)
        binary_target = self.fc3(binary_target)
        binary_target = self.fc4(binary_target)
        return x, y, binary_target

N = 1000
train_input, train_target, train_classes, test_input, test_target, test_classes = generate_pair_sets(N)
train_input, test_input, train_classes, test_classes = preprocess_data(train_input,
                                                                       test_input,
                                                                       train_classes,
                                                                       test_classes)


def train_model(model, train_input1, train_input2, train_target1, train_target2, train_target3,
                mini_batch_size, digit_scalar=1, binary_target_scalar=1):
    """

    :param model: the model that trains our both images with weight sharing
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
    :return: void, the model is trained while calling this method
    """
    criterion = nn.CrossEntropyLoss()
    eta = 1e-1
    optimizer = torch.optim.SGD(model.parameters(), lr=eta, momentum=0)  # check the lectures

    for e in range(25):
        sum_loss = 0
        for b in range(0, train_input1.size(0), mini_batch_size):
            output_x, output_y, output_binary_target = model(train_input1.narrow(0, b, mini_batch_size),
                                                             train_input2.narrow(0, b, mini_batch_size))

            loss_x = criterion(output_x, train_target1.narrow(0, b, mini_batch_size).long())
            loss_y = criterion(output_y, train_target2.narrow(0, b, mini_batch_size).long())
            loss_binary_target = criterion(output_binary_target, train_target3.narrow(0, b, mini_batch_size).long())
            loss = digit_scalar * (loss_x + loss_y) + binary_target_scalar * loss_binary_target
            model.zero_grad()
            loss.backward()
            sum_loss = sum_loss + loss.item()
            optimizer.step()


def train_with_ws(digit_scalar):
    """

    :param digit_scalar: the digit scalar calibrating the auxiliary loss, 0 if we don't want to use auxiliary loss
    :return: void
    """
    print("Preprocessing and setting up the data for training")
    print("----Training the model----")
    if digit_scalar == 0:
        print("----Begin the training without auxiliary loss----")
    else:
        print("Begin the training with auxiliary loss weighted as digit scalar = "+str(digit_scalar))
    model = Net2()
    for k in range(15):
        train_model(model, train_input[0], train_input[1], train_classes[0], train_classes[1], train_target,
                    mini_batch_size, digit_scalar)
        model.eval()
        output1, output2, prediction = model(test_input[0], test_input[1])
        if k == 14:
            print("Accuracy based on classes prediction : ")
            print(compute_error_(compare_and_predict(output1.max(1)[1], output2.max(1)[1]), test_target))
            print("Accuracy based on target prediction")
            print(compute_error_(prediction.max(1)[1], test_target))
            return compute_error_(compare_and_predict(output1.max(1)[1], output2.max(1)[1]), test_target), compute_error_(prediction.max(1)[1], test_target)



