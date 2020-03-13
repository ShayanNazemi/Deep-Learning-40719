import numpy as np
import os
import matplotlib.pyplot as plt
from model import *
from adam import *
from batch_norm import *
from dense import *
from dropout import *
from module import *
from optimizer import *
from relu import *
from sgd import *
from sigmoid import *
from softmax_crossentropy import *

###########################################################################
#                           SGD+Momentum Test                             #
###########################################################################
N, D = 5, 4
np.random.seed(22)
dense = Dense('test', N, D, l2_coef=1.)
dense.dW = np.random.randn(N, D)
dense.db = np.random.randn(D,)

sgd = SGD(1e-2)
sgd.velocities['test'] = {'W': np.random.randn(N, D), 'b': np.zeros_like(dense.b)}

sgd.update(dense)

correct_W = [[-1.04578269, -2.09413292,  1.74896632,  0.23833633],
             [-1.13969324, -0.50236489,  1.28289011, -1.20095538],
             [-0.2181534,  -0.12424752, -1.3418189,   0.13508095],
             [ 0.59827594, -0.35371713, -2.00549095,  3.3314237 ],
             [-1.09332467,  1.15610425,  1.24786695, -1.06838115],]
correct_b = [ 1.86566377, -1.59381353, -0.62684131,  0.33332912]

print('W Relative error:', np.linalg.norm(correct_W - dense.W))
print('b Relative error: ', np.linalg.norm(correct_b - dense.b))

###########################################################################
#                                ADAM Test                                #
###########################################################################
N, D = 5, 4
dense = Dense('test', N, D, l2_coef=1.)
dense.W = np.linspace(-1, 1, N * D).reshape(N, D)
dense.dW = np.linspace(-1, 1, N * D).reshape(N, D)
dense.db = np.zeros(D, )
adam = Adam(1e-2)

m = np.linspace(0.6, 0.9, N * D).reshape(N, D)
v = np.linspace(0.7, 0.5, N * D).reshape(N, D)

adam.m['test'] = {'W': m, 'b': np.zeros(D, )}
adam.v['test'] = {'W': v, 'b': np.zeros(D, )}
adam.iteration_number = 6
adam.update(dense)

next_param = dense.W

correct_next_param = [[-1.00086812, -0.89566086, -0.79045452, -0.68524913],
                      [-0.58004471, -0.47484131, -0.36963895, -0.26443768],
                      [-0.15923753, -0.05403855, 0.05115923, 0.15635575],
                      [0.26155096, 0.36674482, 0.47193728, 0.57712826],
                      [0.68231771, 0.78750557, 0.89269177, 0.99787623]]
correct_v = [[0.7003, 0.68958476, 0.67889169, 0.66822078],
             [0.65757202, 0.64694543, 0.636341, 0.62575873],
             [0.61519861, 0.60466066, 0.59414488, 0.58365125],
             [0.57317978, 0.56273047, 0.55230332, 0.54189834],
             [0.53151551, 0.52115485, 0.51081634, 0.5005]]
correct_m = [[0.44, 0.46473684, 0.48947368, 0.51421053],
             [0.53894737, 0.56368421, 0.58842105, 0.61315789],
             [0.63789474, 0.66263158, 0.68736842, 0.71210526],
             [0.73684211, 0.76157895, 0.78631579, 0.81105263],
             [0.83578947, 0.86052632, 0.88526316, 0.91]]

print('W error: ', np.linalg.norm(correct_next_param - next_param))
print('v error: ', np.linalg.norm(correct_v - adam.v['test']['W']))
print('m error: ', np.linalg.norm(correct_m - adam.m['test']['W']))
