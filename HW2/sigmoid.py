import numpy as np
from module import Module


class Sigmoid(Module):
    def __init__(self, name):
        super(Sigmoid, self).__init__(name)

    def forward(self, x, **kwargs):
        """
        x: input array.
        out: output of Sigmoid function for input x.
        **Save whatever you need for backward pass in self.cache.
        """
        out = 1 / (1 + np.exp(-x))
        self.cache = out
        # todo: implement the forward propagation for Sigmoid module.

        return out

    def backward(self, dout):
        """
        dout: gradients of Loss w.r.t. this layer's output.
        dx: gradients of Loss w.r.t. this layer's input.
        """
        dx = dout * self.cache * (1 - self.cache)
        # todo: implement the backward propagation for Sigmoid module.

        return dx
