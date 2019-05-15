# Important imports
from helper_functions import *
from torch import nn
from torch.nn import functional as F


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
        #self.fc2 = nn.Linear(nb_hidden, 10)
        #self.fc4 = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1_bn(self.conv1(x)), kernel_size=2))
        x = F.relu(F.max_pool2d(self.conv2_bn(self.conv2(x)), kernel_size=2))
        x = F.relu(self.conv3_bn(self.conv3(x)))
        x = self.drop1(x)
        x = F.relu(self.fc1(x.view(-1, 64)))

        x = self.drop2(x)
        #x = self.fc2(x)
        #x = self.fc4(x)
        return x


def classify_digit(x, nb_hidden=100):
    classifier = nn.Sequential(nn.Linear(nb_hidden, 10), nn.Sigmoid())
    return classifier.apply(x)


def classify_binary_target(x, y, nb_hidden=100):
    input = torch.cat([x, y], 1)
    classifier = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(nb_hidden * 2, 2), nn.Sigmoid())
    return classifier.apply(input)


N = 1000


train_input, train_target, train_classes, test_input, test_target, test_classes = generate_pair_sets(N)
train_input, test_input, train_classes, test_classes = preprocess_data(train_input,
                                                                       test_input,
                                                                       train_classes,
                                                                       test_classes)


def train_model(model1, model2, train_input1, train_input2, train_target1, train_target2, train_target3,
                mini_batch_size, classifier1, classifier2, classifier3, digit_scalar=1,
                binary_target_scalar=1):
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


def compute_nb_errors(prediction, target):
    errors = 0
    for (a, b) in zip(prediction, target):
        if a.float() != b.float():
            errors += 1
    return errors / len(prediction) * 100


def train_without_ws(digit_scalar):
    print("Preprocessing and setting up the data for training")
    print("----Training the model----")
    model1 = Net2()
    model2 = Net2()
    classifier1 = nn.Sequential(nn.Linear(100, 10), nn.Sigmoid())
    classifier2 = nn.Sequential(nn.Linear(100, 10), nn.Sigmoid())
    classifier3 = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(100 * 2, 2), nn.Sigmoid())
    if digit_scalar == 0:
        print("----Begin the training without auxiliary loss----")
    else:
        print("Begin the training with auxiliary loss weighted as digit scalar = "+str(digit_scalar))
    for k in range(15):
        train_model(model1, model2, train_input[0], train_input[1], train_classes[0], train_classes[1], train_target,
                    mini_batch_size, classifier1, classifier2, classifier3)
        model1.eval()
        model2.eval()
        encoder1 = model1(test_input[0])
        encoder2 = model2(test_input[1])
        output1 = classifier1(encoder1)
        output2 = classifier2(encoder2)
        prediction = classifier3(torch.cat([encoder1, encoder2], 1))
        if k == 14:
            print("Accuracy based on classes prediction : ")
            print(compute_error_(compare_and_predict(output1.max(1)[1], output2.max(1)[1]), test_target))
            print("Accuracy based on target prediction")
            print(compute_error_(prediction.max(1)[1], test_target))


