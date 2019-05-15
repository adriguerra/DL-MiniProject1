from cnn_with_ws import *
import numpy as np
from cnn_without_ws import *

print("-----Convolution Neural Network with Weight sharing-----")

# Convolution Neural Network with weight sharing and auxiliary loss
#train_with_ws(digit_scalar=1)

# Convolution Neural Network with weight sharing and without auxiliary loss
#train_with_ws(digit_scalar=0)

print("-----Convolution Neural Network without Weight sharing-----")

# Convolution Neural Network without weight sharing and auxiliary loss
#train_without_ws(digit_scalar=1)

# Convolution Neural Network without weight sharing and without auxiliary loss
#train_without_ws(digit_scalar=0)

list_acc_dig_for_std = []
list_acc_ord_for_std = []
for i in range(10):
    a,b = train_without_ws(digit_scalar=1)
    list_acc_dig_for_std.append(a)
    list_acc_ord_for_std.append(b)

print("digit pred std :",np.std(list_acc_dig_for_std))
print("order pred std :",np.std(list_acc_ord_for_std))