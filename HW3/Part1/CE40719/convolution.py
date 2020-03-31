import numpy as np
from CE40719.module import Module


class Convolution(Module):
    def __init__(self, name, filter_shape, stride=1, pad=0, weight_scale=1e-3, l2_coef=.0):
        super(Convolution, self).__init__(name)
        self.W = np.random.normal(scale=weight_scale,
                                  size=filter_shape)  # filter of the layer with shape (F, C, f_h, f_w).
        self.b = np.zeros(filter_shape[0], )  # biases of the layer with shape (F,).
        self.dW = None  # gradients of loss w.r.t. the weights.
        self.db = None  # gradients of loss w.r.t. the biases.
        self.stride = stride
        self.pad = pad
        self.l2_coef = l2_coef
        self.cache = {}

    def forward(self, x, **kwargs):
        """
        x: input array.
        out: output of convolution module for input x.
        Save whatever you need for backward pass in self.cache.
        """
        out = None
        # todo: implement the forward propagation for Dense module.
        (F, C, f_h, f_w) = self.W.shape
        (N, _, i_h, i_w) = x.shape
        x_padded = np.zeros((N, C, f_h + 2 * self.pad, f_w + 2 * self.pad))
        x_padded[:, :, self.pad: i_h + self.pad, self.pad: i_w + self.pad] = x

        o_h = int((i_h + 2 * self.pad - f_h) / self.stride + 1)
        o_w = int((i_w + 2 * self.pad - f_w) / self.stride + 1)

        self.cache['x_padded'] = x_padded
        self.cache['C'] = C

        out = np.zeros([N, F, o_h, o_w])

        for b in range(N):  # for each input
            for f in range(F):  # for each filter of the layer
                for j in range(0, i_h, self.stride):  # in the y direction
                    for i in range(0, i_w, self.stride):  # in the x direction
                        out[b, f, int(j / self.stride), int(i / self.stride)] = np.sum(
                            self.W[f, :, :, :] * x_padded[b, :, j: j + f_h, i: i + f_w]) + self.b[f]
        return out

    def backward(self, dout):
        """
        dout: gradients of Loss w.r.t. this layer's output.
        dx: gradients of Loss w.r.t. this layer's input.
        """
        dx = None
        # todo: implement the backward propagation for Dense module.
        x_padded = self.cache['x_padded']

        (F, C, f_h, f_w) = self.W.shape
        (N, _, i_h, i_w) = x_padded.shape
        (_, _, grad_h, grad_w) = dout.shape

        dx = np.zeros(x_padded.shape)
        self.dW = np.zeros(self.W.shape)
        self.db = np.zeros(self.b.shape)

        for b in range(N):  # for each input
            for f in range(F):  # for each filter of the layer
                for j in range(grad_h):  # in the y direction
                    for i in range(grad_w):  # in the x direction
                        dx[b, :, j * self.stride: j * self.stride + f_h, i * self.stride: i * self.stride + f_w] += self.W[f] * dout[b][f][j][i]

                        self.dW[f] += dout[b][f][j][i] * x_padded[b, :, j * self.stride: j * self.stride + f_h, i * self.stride: i * self.stride + f_w]
                        self.db[f] += dout[b][f][j][i]

        dx = dx[:, :, self.pad:-self.pad, self.pad:-self.pad]
        # don't forget to update self.dW and self.db.

        return dx
