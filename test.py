from cnn_with_ws import *
from cnn_without_ws import *

print("-----Convolution Neural Network with Weight sharing-----")

# Convolution Neural Network with weight sharing and auxiliary loss
train_with_ws(digit_scalar=1)

# Convolution Neural Network with weight sharing and without auxiliary loss
train_with_ws(digit_scalar=0)

print("-----Convolution Neural Network without Weight sharing-----")

# Convolution Neural Network without weight sharing and auxiliary loss
train_without_ws(digit_scalar=1)

# Convolution Neural Network without weight sharing and without auxiliary loss
train_without_ws(digit_scalar=0)
