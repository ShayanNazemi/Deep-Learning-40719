import numpy as np
from CE40719.module import Module


class Convolution(Module):
    def __init__(self, name, filter_shape, stride = 1, pad =0, weight_scale =1e-3, l2_coef=.0):
        super(Convolution, self).__init__(name)
        self.W = np.random.normal(scale = weight_scale, size = filter_shape)  # filter of the layer with shape (F, C, f_h, f_w).
        self.b = np.zeros(filter_shape[0], )  # biases of the layer with shape (F,).
        self.dW = None  # gradients of loss w.r.t. the weights.
        self.db = None  # gradients of loss w.r.t. the biases.
        self.stride = stride
        self.pad = pad
        self.l2_coef = l2_coef

    def forward(self, x, **kwargs):
        """
        x: input array.
        out: output of convolution module for input x.
        Save whatever you need for backward pass in self.cache.
        """
        out = None
        # todo: implement the forward propagation for Dense module.
        
        return out

    def backward(self, dout):
        """
        dout: gradients of Loss w.r.t. this layer's output.
        dx: gradients of Loss w.r.t. this layer's input.
        """
        dx = None
        # todo: implement the backward propagation for Dense module.
        # don't forget to update self.dW and self.db.
        
        return dx

