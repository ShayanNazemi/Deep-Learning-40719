import numpy as np
from module import Module
from dense import *


class BatchNormalization(Module):
    def __init__(self, name, input_shape, momentum=.9, epsilon=1e-5):
        super(BatchNormalization, self).__init__(name)
        self.momentum = momentum  # momentum rate for computing running_mean and running_var
        self.gamma = np.random.randn(input_shape)  # gamma parameters for batch norm.
        self.beta = np.random.randn(input_shape)  # beta parameters for batch norm.
        self.eps = epsilon  # this parameter will be used to avoid division by zero!

        self.running_mean = np.zeros(input_shape)  # mean for test phase
        self.running_var = np.zeros(input_shape)  # var for test phase

        self.dbeta = 0  # gradients of loss w.r.t. the beta parameters.
        self.dgamma = 0  # gradients of loss w.r.t. the gamma parameters.

    def forward(self, x, **kwargs):
        """
        x: input array.
        out: output of Dense module for input x.
        **Save whatever you need for backward pass in self.cache.
        """

        if self.phase == 'Train':
            # todo: implement the forward propagation for batch norm module for train phase.
            # and update running_mean and running_var using sample_mean and sample_var for test phase.
            (B, _) = x.shape
            miu = np.mean(x, axis=0)
            var = np.var(x, axis=0)
            centered = x - miu
            normalized = centered / np.sqrt(var + self.eps)
            out = self.gamma * normalized + self.beta

            self.running_mean = self.running_mean * self.momentum + (1 - self.momentum) * miu
            self.running_var = self.running_var * self.momentum + (1 - self.momentum) * var

            self.cache = (B, normalized, centered, var)

            return out
        else:
            # todo: implement the forward propagation for batch norm module for test phase.
            # use running_mean and running_var
            out = (x - self.running_mean[None, :]) / np.sqrt(self.running_var + self.eps)
            out *= self.gamma
            out += self.beta
            return out

    def backward(self, dout):
        """
         dout: gradients of Loss w.r.t. this layer's output.
         dx: gradients of Loss w.r.t. this layer's input.
         """
        # todo: implement the backward propagation for batch norm module.(train phase only)
        # don't forget to update self.dgamma and self.dbeta.
        dx = None

        (B, normalized, centered, var) = self.cache

        dgamma = (dout * normalized).sum(axis=0)
        dbeta = dout.sum(axis=0)
        dnorm = dout * self.gamma

        temp = 1 / np.sqrt(var + self.eps)

        dx = temp * dnorm - temp / B * dnorm.sum(axis=0) - temp * (dnorm * normalized).sum(axis=0) * normalized / B

        return dx

    def reset(self):
        self.running_mean = np.zeros_like(self.running_mean)
        self.running_var = np.zeros_like(self.running_var)
