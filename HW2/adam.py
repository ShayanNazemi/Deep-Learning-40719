import numpy as np
from optimizer import Optimizer
from dense import Dense
from batch_norm import BatchNormalization


class Adam(Optimizer):
    def __init__(self, learning_rate=1e-2, beta1=.9, beta2=.999, epsilon=1e-8):
        super(Adam, self).__init__(learning_rate)
        self.beta1 = beta1  # rate for Moving average of gradient.(m)
        self.beta2 = beta2  # rate for Moving average of squared gradient.(v)
        self.eps = epsilon  # this parameter will be used to avoid division by zero!
        self.v = {}  # Moving average of squared gradients.
        self.m = {}  # Moving average of gradients.
        # m and v are dictionaries that for each module save a dictionary by their names.
        # like {module_name: m_dict} or {module_name: v_dict}
        # for Dense modules v_dict and m_dict are dictionaries like {"W": v of W, "b": v of b} 

    #	or {"W": m of W, "b": m of b}
    # for Batch Norm modules v_dict and m_dict are dictionaries like {"gamma": v of gamma, "beta": v of beta}
    #       or {"gamma": m of gamma, "beta": m of beta}

    def update(self, module):
        if not (isinstance(module, Dense) or isinstance(module, BatchNormalization)):
            return  # the only modules that contain trainable parameters are dense and batch norm.

        # todo: implement adam update rules for both Dense and Batch Norm modules.
        params = None
        if isinstance(module, Dense):
            params = ('W', 'b')
            try:
                temp = self.m[module.name]['W']
            except:
                self.m[module.name] = {'W' : np.zeros_like(module.W), 'b' : np.zeros_like(module.b)}
                self.v[module.name] = {'W' : np.zeros_like(module.W), 'b' : np.zeros_like(module.b)}

            self.m[module.name]['W'] *= self.beta1
            self.m[module.name]['W'] += (1 - self.beta1) * module.dW
            self.v[module.name]['W'] *= self.beta2
            self.v[module.name]['W'] += (1 - self.beta2) * (module.dW ** 2)

            self.m[module.name]['b'] *= self.beta1
            self.m[module.name]['b'] += (1 - self.beta1) * module.db
            self.v[module.name]['b'] *= self.beta2
            self.v[module.name]['b'] += (1 - self.beta2) * (module.db ** 2)

            m_hat_w = self.m[module.name]['W'] / (1 - self.beta1 ** self.iteration_number)
            v_hat_w = self.v[module.name]['W'] / (1 - self.beta2 ** self.iteration_number)
            m_hat_b = self.m[module.name]['b'] / (1 - self.beta1 ** self.iteration_number)
            v_hat_b = self.v[module.name]['b'] / (1 - self.beta2 ** self.iteration_number)

            module.W -= self.learning_rate / np.sqrt(self.eps + v_hat_w) * m_hat_w
            module.b -= self.learning_rate / np.sqrt(self.eps + v_hat_b) * m_hat_b

        elif isinstance(module, BatchNormalization):
            params = ('gamma', 'beta')
            try:
                temp = self.m[module.name]['gamma']
            except:
                self.m[module.name] = {'gamma' : np.zeros_like(module.gamma), 'beta' : np.zeros_like(module.beta)}
                self.v[module.name] = {'gamma' : np.zeros_like(module.gamma), 'beta' : np.zeros_like(module.beta)}

            self.m[module.name]['gamma'] *= self.beta1
            self.m[module.name]['gamma'] += (1 - self.beta1) * module.dgamma
            self.v[module.name]['gamma'] *= self.beta2
            self.v[module.name]['gamma'] += (1 - self.beta2) * (module.dgamma ** 2)

            self.m[module.name]['beta'] *= self.beta1
            self.m[module.name]['beta'] += (1 - self.beta1) * module.dbeta
            self.v[module.name]['beta'] *= self.beta2
            self.v[module.name]['beta'] += (1 - self.beta2) * (module.dbeta ** 2)

            m_hat_gamma = self.m[module.name]['gamma'] / (1 - self.beta1 ** self.iteration_number)
            v_hat_gamma = self.v[module.name]['gamma'] / (1 - self.beta2 ** self.iteration_number)
            m_hat_beta = self.m[module.name]['beta'] / (1 - self.beta1 ** self.iteration_number)
            v_hat_beta = self.v[module.name]['beta'] / (1 - self.beta2 ** self.iteration_number)

            module.gamma -= self.learning_rate / np.sqrt(self.eps + v_hat_gamma) * m_hat_gamma
            module.beta -= self.learning_rate / np.sqrt(self.eps + v_hat_beta) * m_hat_beta
