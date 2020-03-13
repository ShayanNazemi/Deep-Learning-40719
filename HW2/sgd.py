import numpy as np
from optimizer import Optimizer
from dense import Dense
from batch_norm import BatchNormalization


class SGD(Optimizer):
    def __init__(self, learning_rate=1e-2, momentum=.9):
        super(SGD, self).__init__(learning_rate)
        self.momentum = momentum  # momentum rate
        self.velocities = {}  # a moving average of the gradients.
        # velocities is a dictionary that saves a dictionary for each module by its name.
        # {module_name: v_dict}
        # for Dense modules v_dict is a dictionary like {"W": velocity of W, "b": velocity of b}
        # for Batch Norm modules v_dict is a dictionary like {"gamma": velocity of gamma, "beta": velocity of beta}

    def update(self, module):
        if not (isinstance(module, Dense) or isinstance(module, BatchNormalization)):
            return  # the only modules that contain trainable parameters are dense and batch norm.

        # todo: implement sgd + momentum update rules for both Dense and Batch Norm modules.
        params = None
        if isinstance(module, Dense):
            params = ('W', 'b')
            try:
                temp = self.velocities[module.name]['W']
            except:
                self.velocities[module.name] = {'W': np.zeros_like(module.W), 'b': np.zeros_like(module.b)}
                self.velocities[module.name] = {'W': np.zeros_like(module.W), 'b': np.zeros_like(module.b)}

            self.velocities[module.name]['W'] *= self.momentum
            self.velocities[module.name]['W'] -= self.learning_rate * module.dW

            self.velocities[module.name]['b'] *= self.momentum
            self.velocities[module.name]['b'] -= self.learning_rate * module.db

            module.W += self.velocities[module.name]['W']
            module.b += self.velocities[module.name]['b']

        elif isinstance(module, BatchNormalization):
            params = ('gamma', 'beta')
            try:
                temp = self.velocities[module.name]['gamma']
            except:
                self.velocities[module.name] = {'gamma': np.zeros_like(module.gamma),
                                                'beta': np.zeros_like(module.beta)}
                self.velocities[module.name] = {'gamma': np.zeros_like(module.gamma),
                                                'beta': np.zeros_like(module.beta)}

            self.velocities[module.name]['gamma'] *= self.momentum
            self.velocities[module.name]['gamma'] -= self.learning_rate * module.dgamma

            self.velocities[module.name]['beta'] *= self.momentum
            self.velocities[module.name]['beta'] -= self.learning_rate * module.dbeta

            module.gamma += self.velocities[module.name]['gamma']
            module.beta += self.velocities[module.name]['beta']
