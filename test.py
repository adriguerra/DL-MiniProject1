from cnn_with_ws import *
from cnn_without_ws import *

print("-----CNN with Weight sharing-----")

# Convolutional Neural Network with weight sharing and auxiliary loss
train_with_ws(digit_scalar=1)

# Convolutional Neural Network with weight sharing and without auxiliary loss
train_with_ws(digit_scalar=0)

print("-----CNN without Weight sharing-----")

# Convolutional Neural Network without weight sharing and auxiliary loss
train_without_ws(digit_scalar=1)

# Convolutional Neural Network without weight sharing and without auxiliary loss
train_without_ws(digit_scalar=0)
