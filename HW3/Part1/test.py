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
#            Group-Normalization backward Test                            #
###########################################################################
np.random.seed(40959)
N, C, H, W = 2, 4, 2, 2
G = 2
x = 5 * np.random.randn(N, C, H, W) + 12
dout = np.random.randn(N, C, H, W)
norm = GroupBatchNorm('test',(N, C, H, W), G)
_ = norm.forward(x)
dx = norm.backward(dout)
dgamma = norm.dgamma
dbeta = norm.dbeta
correct_dx = np.array([[[[-0.34315175, -0.12381128],[ 0.03543989, -0.10811583]],
                        [[ 0.23704048,  0.26681368],[-0.35225047,  0.38803527]],
                        [[ 0.08007765, -0.00905627],[-0.05405272, -0.10643134]],
                        [[-0.09270617, -0.26378193],[ 0.36204868,  0.08390209]]],
                       [[[-0.15694482,  0.13000584],[-0.10158775, -0.07278944]],
                        [[ 0.29256627,  0.00665728],[-0.08065789, -0.0172495 ]],
                        [[-0.01647735,  0.09978986],[-0.11498191,  0.05169455]],
                        [[ 0.09743741, -0.18717901],[ 0.12091729, -0.05120085]]]])
print('Relative error dx:', np.linalg.norm(dx - correct_dx))
correct_dgamma = np.array([[[[-3.28150076]],[[-5.58372413]],[[ 2.98735869]],[[ 3.02274058]]]])
print('Relative error dgamma:', np.linalg.norm(dgamma - correct_dgamma))
correct_dbeta = np.array([[[[ 0.28621793]],[[ 5.8519097 ]],[[-3.48103948]],[[-2.95280322]]]])
print('Relative error dbeta:', np.linalg.norm(dbeta - correct_dbeta))