# -*- coding: utf-8 -*-
from collections import OrderedDict
from abc import ABCMeta, abstractmethod

import numpy as np

import theano
import theano.tensor as tensor

from .defaults import FLOAT
from .nmtutils import unzip


def get_optimizer(name):
    optimizers = {
            'sgd'       : SGD,
            'adam'      : Adam,
            'rmsprop'   : RMSProp,
            'adadelta'  : Adadelta,
            }
    return optimizers[name]

class Optimizer(object, metaclass=ABCMeta):
    def __init__(self, lr0):
        # Learning rate shared variable
        self.lr = theano.shared(np.float64(lr0).astype(FLOAT), name='lrate')

        # Theano shared variables for accumulator tensors
        self.history = OrderedDict()

        # Store grad variables given with get_updates()
        self.grads = None

        # Gradient noise per update
        self.grad_noise_factor = 0.

    def set_trng(self, trng):
        """Save Theano RNG."""
        self.trng = trng

    def set_gradient_noise(self, factor):
        """Set gradient noise factor."""
        self.grad_noise_factor = factor

    def init_value(self, shape, name, history=None):
        """Initialize a variable with zero or last value."""
        value = history[name] if history else np.zeros(shape, dtype=FLOAT)

        # Create the shared variable and store it
        self.history[name] = theano.shared(value, name)
        return self.history[name]

    def get_history(self):
        """Returns a dictionary of numpy tensors for history variables."""
        return unzip(self.history)

    def set_lrate(self, lrate):
        """Update the internal lrate."""
        self.lr.set_value(lrate)

    @abstractmethod
    def get_updates(self, tparams, grads, history=None):
        """Return updates list for params."""
        pass


#############################
# Stochastic Gradient Descent
#############################
class SGD(Optimizer):
    def __init__(self, lr0=0.01):
        super(SGD, self).__init__(lr0)

    def get_updates(self, tparams, grads, history=None):
        self.grads = grads
        updates = []
        for tparam, grad in zip(tparams.values(), grads):
            updates.append((tparam, tparam - self.lr * grad))

        return updates

#########
# RMSProp
#########
class RMSProp(Optimizer):
    def __init__(self, lr0=0.001, rho=0.95, eps=1e-6):
        super(RMSProp, self).__init__(lr0)
        self.rho = rho
        self.eps = eps

    def get_updates(self, tparams, grads, history=None):
        self.grads = grads
        updates = []
        for tparam, grad in zip(tparams.values(), grads):
            # Accumulate gradient squares
            v = self.init_value(tparam.get_value().shape, '%s_v' % tparam.name, history)

            # rho * past + (1 - rho) * current
            v_new = (self.rho * v) + (1. - self.rho) * grad**2

            updates.append((v, v_new))
            updates.append((tparam, tparam - (self.lr * grad / tensor.sqrt(v_new + self.eps))))

        return updates

##########
# Adadelta
##########
class Adadelta(Optimizer):
    def __init__(self, lr0=1., rho=0.95, eps=1e-6):
        super(Adadelta, self).__init__(lr0)
        self.rho = rho
        self.eps = eps

    def get_updates(self, tparams, grads, history=None):
        self.grads = grads
        updates = []
        for tparam, grad in zip(tparams.values(), grads):
            v = self.init_value(tparam.get_value().shape, '%s_v' % tparam.name, history)
            u = self.init_value(tparam.get_value().shape, '%s_u' % tparam.name, history)

            # Accumulate gradient squares
            # rho * past + (1 - rho) * current
            v_new = (self.rho * v) + (1. - self.rho) * grad**2
            updates.append((v, v_new))

            # Update rule
            up = (grad * tensor.sqrt(u + self.eps) / tensor.sqrt(v_new + self.eps))
            updates.append((tparam, tparam - self.lr * up))

            # Accumulate update magnitudes
            updates.append((u, self.rho * u + (1. - self.rho) * up**2))

        return updates

######
# Adam
######
class Adam(Optimizer):
    def __init__(self, *args, lr0=0.0001, b1=0.9, b2=0.999, eps=1e-8):
        super().__init__(lr0)
        self.b1  = b1
        self.b2  = b2
        self.eps = eps

    def get_updates(self, tparams, grads, history=None):
        self.grads = grads
        updates = []

        # Iteration counter, 'None' for shape creates a scalar
        i = self.init_value(None, 'i', history)

        i_t = i + 1.

        # Running learning-rate that will eventually -> lr0
        lr_t = self.lr * (tensor.sqrt(1. - self.b2**i_t) / (1. - self.b1**i_t))

        # Increment iteration counter
        updates.append((i, i_t))

        for tparam, grad in zip(tparams.values(), grads):
            m = self.init_value(tparam.get_value().shape, '%s_m' % tparam.name, history)
            v = self.init_value(tparam.get_value().shape, '%s_v' % tparam.name, history)

            if self.grad_noise_factor > 0:
                # Sample normal noise from N(0, sqrt(factor/((1+t)**0.55))).
                var = self.grad_noise_factor / (i_t**0.55)
                noise = self.trng.normal(grad.shape, std=tensor.sqrt(stdev), dtype=FLOAT)
                grad += noise

            m_t = (self.b1 * m) + ((1. - self.b1) * grad)
            v_t = (self.b2 * v) + ((1. - self.b2) * grad**2)
            p_t = tparam - (lr_t * (m_t / (tensor.sqrt(v_t) + self.eps)))

            # Add updates
            updates.append((m, m_t))
            updates.append((v, v_t))
            updates.append((tparam, p_t))

        return updates
