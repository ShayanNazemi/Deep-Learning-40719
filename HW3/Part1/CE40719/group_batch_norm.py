import numpy as np
from CE40719.module import Module


class GroupBatchNorm(Module):
    def __init__(self, name, input_shape, G=1, epsilon=1e-5):
        super(GroupBatchNorm, self).__init__(name)
        (N, C, H, W) = input_shape
        self.gamma = np.ones((1, C, 1, 1))  # Scale parameter, of shape (C,)
        self.beta = np.zeros((1, C, 1, 1))  # Shift parameter, of shape (C,)
        self.G = G  # Integer number of groups to split into, should be a divisor of C
        self.eps = epsilon
        self.dbeta = 0
        self.dgamma = 0
        self.cache = None

    def forward(self, x, **kwargs):
        """
        Computes the forward pass for spatial group normalization.
        In contrast to layer normalization, group normalization splits each entry 
        in the data into G contiguous pieces, which it then normalizes independently.
        Per feature shifting and scaling are then applied to the data, in a manner 
        identical to that of batch normalization.
        **Save whatever you need for backward pass in self.cache.
        """
        out = None
        # TODO: Implement the forward pass for spatial group normalization.

        (N, C, H, W) = x.shape
        x_ = np.reshape(x, (N, self.G, C // self.G, H, W))
        mean = np.mean(x_, (2, 3, 4), keepdims=True)
        var = np.var(x_, (2, 3, 4), keepdims=True)
        out = (x_ - mean) / np.sqrt(var + self.eps)

        self.cache = (x_, mean, var, out.reshape((N, C, H, W)))

        out = out.reshape((N, C, H, W)) * self.gamma + self.beta

        return out

    def backward(self, dout):
        """
        Computes the backward pass for spatial group normalization.
        dx: Gradient with respect to inputs, of shape (N, C, H, W)
        dgamma: Gradient with respect to scale parameter, of shape (C,)
        dbeta: Gradient with respect to shift parameter, of shape (C,)
        """
        dx = dout
        # TODO: Implement the backward pass for spatial group normalization.
        # don't forget to update self.dgamma and self.dbeta.

        (x, mean, var, norm) = self.cache
        (N, C, H, W) = dout.shape

        dnorm = (dout * self.gamma).reshape((N, self.G, C // self.G, H, W))
        self.dbeta = np.sum(dout, axis=(0, 2, 3), keepdims=True)
        self.dgamma = np.sum(dout * norm, axis=(0, 2, 3), keepdims=True)

        dx = (dnorm - (dnorm.mean(axis=(2, 3, 4), keepdims=True) + (dnorm * norm.reshape(dnorm.shape)).mean(axis=(2, 3, 4), keepdims=True) * norm.reshape(dnorm.shape))) / np.sqrt(var + self.eps)
        dx = dx.reshape(N, C, H, W)

        return dx
