import numpy as np
from CE40719.module import Module


class MaxPool(Module):
    def __init__(self, name, height=1, width=1, stride=1):
        super(MaxPool, self).__init__(name)
        self.height = height  # The height of each pooling region
        self.width = width  # The width of each pooling region
        self.stride = stride  # The distance between adjacent pooling regions

    def forward(self, x, **kwargs):
        """
        x: input array.
        out: output of max pool module for input x.
        Save whatever you need for backward pass in self.cache.
        """
        out = None
        # todo: implement the forward propagation for max_pool module.



        return out

    def backward(self, dout):
        """
        dout: gradients of Loss w.r.t. this layer's output.
        dx: gradients of Loss w.r.t. this layer's input.
        """
        dx = None
        # todo: implement the backward propagation for Dense module.

        return dx
