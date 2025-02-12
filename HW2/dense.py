import numpy as np
from module import Module


class Dense(Module):
    def __init__(self, name, input_dim, output_dim, l2_coef=.0):
        super(Dense, self).__init__(name)

        self.l2_coef = l2_coef  # coefficient of l2 regularization.

        self.W = np.random.randn(input_dim, output_dim)  # weights of the layer.
        self.b = np.random.randn(output_dim, )  # biases of the layer.
        self.dW = None  # gradients of loss w.r.t. the weights.
        self.db = None  # gradients of loss w.r.t. the biases.

    def forward(self, x, **kwargs):
        """
        x: input array.
        out: output of Dense module for input x.
        **Save whatever you need for backward pass in self.cache.
        """
        out = np.matmul(x, self.W) + self.b
        self.cache = x
        # todo: implement the forward propagation for Dense module.

        return out

    def backward(self, dout):
        """
        dout: gradients of Loss w.r.t. this layer's output.
        dx: gradients of Loss w.r.t. this layer's input.
        """
        x = self.cache
        self.db = dout.sum(axis=0)
        dx = np.matmul(dout, self.W.T)
        self.dW = np.matmul(x.T, dout) + 2 * self.l2_coef * self.W

        # todo: implement the backward propagation for Dense module.
        # don't forget to update self.dW and self.db.

        return dx
