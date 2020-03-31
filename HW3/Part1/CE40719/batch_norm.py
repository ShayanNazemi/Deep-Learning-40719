import numpy as np
from module import Module


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

            out = None

            return out
        else:
            # todo: implement the forward propagation for batch norm module for test phase.
            # use running_mean and running_var
            out = None

            return out

    def backward(self, dout):
        """
         dout: gradients of Loss w.r.t. this layer's output.
         dx: gradients of Loss w.r.t. this layer's input.
         """
        # todo: implement the backward propagation for batch norm module.(train phase only)
        # don't forget to update self.dgamma and self.dbeta.
        dx = None

        return dx

    def reset(self):
        self.running_mean = np.zeros_like(self.running_mean)
        self.running_var = np.zeros_like(self.running_var)
