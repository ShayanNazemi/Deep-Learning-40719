import numpy as np
from module import Module


class SoftmaxCrossentropy(Module):
    def __init__(self, name):
        super(SoftmaxCrossentropy, self).__init__(name)

    def forward(self, x, **kwargs):
        y = kwargs.pop('y', None)
        """
        x: input array.
        y: real labels for this input.
        probs: probabilities of labels for this input.
        loss: cross entropy loss between probs and real labels.
        **Save whatever you need for backward pass in self.cache.
        """

        (N, _) = x.shape
        temp = np.exp(x - np.max(x))
        probs = temp / np.sum(temp, axis=1)[:, None]
        loss = np.sum(-np.log(probs[range(N), y])) / N
        self.cache = {'probs': probs, 'N': N, 'y': y}
        # todo: implement the forward propagation for probs and compute cross entropy loss
        # NOTE: implement a numerically stable version.If you are not careful here
        # it is easy to run into numeric instability!

        return loss, probs

    def backward(self, dout=0):
        dx = dout
        probs = self.cache['probs']
        N = self.cache['N']
        probs[range(N), self.cache['y']] -= 1
        dx = (probs/N)


        # todo: implement the backward propagation for this layer.

        return dx
