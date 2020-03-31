import numpy as np
import os
import matplotlib.pyplot as plt
from CE40719.module import *
from CE40719.batch_norm import *
from CE40719.cnn_batch_norm import *
from CE40719.convolution import *
from CE40719.group_batch_norm import *
from CE40719.max_pool import *

###########################################################################
#                            Max pool test                                #
###########################################################################
np.random.seed(40959)
x_shape = (2, 3, 4, 4)
x = np.linspace(-0.3, 0.4, num=np.prod(x_shape)).reshape(x_shape)
pool = MaxPool('test', height=2, width=2, stride=2)
output = pool.forward(x)

correct_output = np.array([[[[-0.26315789, -0.24842105],
                             [-0.20421053, -0.18947368]],
                            [[-0.14526316, -0.13052632],
                             [-0.08631579, -0.07157895]],
                            [[-0.02736842, -0.01263158],
                             [0.03157895, 0.04631579]]],
                           [[[0.09052632, 0.10526316],
                             [0.14947368, 0.16421053]],
                            [[0.20842105, 0.22315789],
                             [0.26736842, 0.28210526]],
                            [[0.32631579, 0.34105263],
                             [0.38526316, 0.4]]]])

print('Relative error forward pass:', np.linalg.norm(output - correct_output))

x = np.random.randn(3, 2, 2, 2)
dout = np.random.randn(3, 2, 1, 1)
pool = MaxPool('test', height=2, width=2, stride=2)
out = pool.forward(x)
dx = pool.backward(dout)
correct_dx = np.array([[[[0., 1.21236066], [0., 0.]],
                        [[0.45107133, 0.], [0., 0.]]],
                       [[[0., 0.], [-0.86463156, 0.]],
                        [[0., 0.], [0., -0.39180953]]],
                       [[[0.93694169, 0.], [0., 0.]],
                        [[0., 0.], [-0.08002411, 0.]]]])
print('Relative error dx:', np.linalg.norm(dx - correct_dx))
