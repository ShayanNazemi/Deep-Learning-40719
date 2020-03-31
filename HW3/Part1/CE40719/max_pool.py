import numpy as np
from CE40719.module import Module


class MaxPool(Module):
    def __init__(self, name, height=1, width=1, stride=1):
        super(MaxPool, self).__init__(name)
        self.height = height  # The height of each pooling region
        self.width = width  # The width of each pooling region
        self.stride = stride  # The distance between adjacent pooling regions
        self.cache = {}

    def forward(self, x, **kwargs):
        """
        x: input array.
        out: output of max pool module for input x.
        Save whatever you need for backward pass in self.cache.
        """
        out = None
        # todo: implement the forward propagation for max_pool module.

        (N, C, i_h, i_w) = x.shape
        (s, h, w) = (self.stride, self.height, self.width)
        (o_h, o_w) = (int(np.floor((i_h - h) / s) + 1), int(np.floor((i_w - w) / s) + 1))

        out = np.zeros((N, C, o_h, o_w))
        max_indices = np.zeros((N, C, o_h, o_w))

        for b in range(N):
            for c in range(C):
                for j in range(0, o_h):
                    for i in range(0, o_w):
                        pool = x[b, c, j*s: j*s + h, i*s: i*s + w]

                        out[b, c, j, i] = np.max(pool)
                        max_indices[b, c, j, i] = np.argmax(pool)

        self.cache['indices'] = max_indices
        self.cache['x_shape'] = (N, C, i_h, i_w)

        return out

    def backward(self, dout):
        """
        dout: gradients of Loss w.r.t. this layer's output.
        dx: gradients of Loss w.r.t. this layer's input.
        """
        dx = None
        # todo: implement the backward propagation for Dense module.
        (N, C, i_h, i_w) = self.cache['x_shape']
        (s, h, w) = (self.stride, self.height, self.width)
        (_, _, o_h, o_w) = dout.shape

        dx = np.zeros(self.cache['x_shape'])

        for b in range(N):
            for c in range(C):
                for j in range(0, o_h):
                    for i in range(0, o_w):
                        index = self.cache['indices'][b, c, j, i]
                        dx[b, c, j * s + int(index / h), i * s + int(index % w)] = dout[b, c, j, i]
        return dx
