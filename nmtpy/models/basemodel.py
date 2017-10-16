# -*- coding: utf-8 -*-
from collections import OrderedDict

from abc import ABCMeta, abstractmethod

import theano
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import numpy as np
from ..nmtutils import unzip, get_param_dict
from ..sysutils import readable_size, get_temp_file, get_valid_evaluation
from ..defaults import INT, FLOAT
from ..optimizers import get_optimizer

class BaseModel(object, metaclass=ABCMeta):
    def __init__(self, **kwargs):
        # This will save all arguments as instance attributes
        self.__dict__.update(kwargs)

        # Will be set when set_dropout is first called
        self._use_dropout   = None

        # Theano TRNG
        self._trng          = None

        # Input tensor lists
        self.inputs         = None

        # Theano variables
        self.f_log_probs    = None
        self.f_init         = None
        self.f_next         = None

        # Model parameters, i.e. weights and biases
        self.initial_params = None
        self.tparams        = None

        # Iterators
        self.train_iterator = None
        self.valid_iterator = None

        # A theano shared variable for lrate annealing
        self.learning_rate  = None

        # Optimizer instance (will not be serialized)
        self.__opt          = None

    @staticmethod
    def beam_search(inputs, f_inits, f_nexts, beam_size=12, maxlen=100, suppress_unks=False, **kwargs):
        # Override this from your classes
        pass

    def set_options(self, optdict):
        """Filter out None's and '__[a-zA-Z]' then store into self._options."""
        self._options = OrderedDict()
        for k,v in optdict.items():
            # Don't keep model attributes with _ prefix
            if v is not None and not k.startswith('_'):
                self._options[k] = v

    def set_trng(self, seed):
        """Set the seed for Theano RNG."""
        if seed == 0:
            # No seed given, randomly pick the seed
            seed = np.random.randint(2**29) + 1
        self._trng = RandomStreams(seed)

    def set_dropout(self, val):
        """Set dropout indicator for activation scaling if dropout is available through configuration."""
        if self._use_dropout is None:
            self._use_dropout = theano.shared(np.float64(0.).astype(FLOAT))
        else:
            self._use_dropout.set_value(float(val))

    def update_lrate(self, lrate):
        """Update learning rate."""
        self.__opt.set_lrate(lrate)

    def get_nb_params(self):
        """Return the number of parameters of the model."""
        return readable_size(sum([p.size for p in self.initial_params.values()]))

    def save(self, fname):
        """Save model parameters as .npz."""
        if self.tparams is not None:
            np.savez(fname, tparams=unzip(self.tparams), opts=self._options)
        else:
            np.savez(fname, opts=self._options)

    def load(self, params):
        """Restore .npz checkpoint file into model."""
        self.tparams = OrderedDict()

        if isinstance(params, str):
            # Filename, load from it
            params = get_param_dict(params)

        for k,v in params.items():
            self.tparams[k] = theano.shared(v.astype(FLOAT), name=k)

    def init_shared_variables(self):
        """Initialize the shared variables of the model."""
        # Create tensor dict
        self.tparams = OrderedDict()

        # Fill it with initial random weights
        for kk, pp in self.initial_params.items():
            self.tparams[kk] = theano.shared(pp, name=kk)

    def update_shared_variables(self, _from):
        """Reset some variables from _from dict."""
        for kk in _from.keys():
            self.tparams[kk].set_value(_from[kk])

    def val_loss(self, mean=True):
        """Compute validation loss."""
        probs = []

        # dict of x, x_mask, y, y_mask
        for data in self.valid_iterator:
            # Don't fail if data doesn't contain y_mask. The loss won't
            # be normalized but the training will continue
            norm = data['y_mask'].sum(0) if 'y_mask' in data else 1
            log_probs = self.f_log_probs(*list(data.values())) / norm
            probs.extend(log_probs)

        if mean:
            return np.array(probs).mean()
        else:
            return np.array(probs)

    def get_l2_weight_decay(self, decay_c, skip_bias=True):
        """Return l2 weight decay regularization term."""
        decay_c = theano.shared(np.float64(decay_c).astype(FLOAT), name='decay_c')
        weight_decay = 0.
        for kk, vv in self.tparams.items():
            # Skip biases for L2 regularization
            if not skip_bias or (skip_bias and vv.get_value().ndim > 1):
                weight_decay += (vv ** 2).sum()
        weight_decay *= decay_c
        return weight_decay

    def get_clipped_grads(self, grads, clip_c):
        """Clip gradients a la Pascanu et al."""
        g2 = 0.
        new_grads = []
        for g in grads:
            g2 += (g**2).sum()
        for g in grads:
            new_grads.append(tensor.switch(g2 > (clip_c**2),
                                           g / tensor.sqrt(g2) * clip_c,
                                           g))
        return new_grads

    def build_optimizer(self, cost, regcost, clip_c, dont_update=None, opt_history=None):
        """Build optimizer by optionally disabling learning for some weights."""
        tparams = OrderedDict(self.tparams)

        # Filter out weights that we do not want to update during backprop
        if dont_update is not None:
            for key in list(tparams.keys()):
                if key in dont_update:
                    del tparams[key]

        # Our final cost
        final_cost = cost.mean()

        # If we have a regularization cost, add it
        if regcost is not None:
            final_cost += regcost

        # Normalize cost w.r.t sentence lengths to correctly compute perplexity
        # Only active when y_mask is available
        if 'y_mask' in self.inputs:
            norm_cost = (cost / self.inputs['y_mask'].sum(0)).mean()
            if regcost is not None:
                norm_cost += regcost
        else:
            norm_cost = final_cost

        # Get gradients of cost with respect to variables
        # This uses final_cost which is not normalized w.r.t sentence lengths
        grads = tensor.grad(final_cost, wrt=list(tparams.values()))

        # Clip gradients if requested
        if clip_c > 0:
            grads = self.get_clipped_grads(grads, clip_c)

        # Create optimizer, self.lrate is passed from nmt-train
        self.__opt = get_optimizer(self.optimizer)(lr0=self.lrate)
        self.__opt.set_trng(self._trng)
        #TODO: parameterize this! self.__opt.set_gradient_noise(0.1)

        # Get updates
        updates = self.__opt.get_updates(tparams, grads, opt_history)

        # Compile forward/backward function
        self.train_batch = theano.function(list(self.inputs.values()), norm_cost, updates=updates)

    def run_beam_search(self, beam_size=12, n_jobs=8, metric='bleu', f_valid_out=None):
        """Save model under /tmp for passing it to nmt-translate."""
        # Save model temporarily
        with get_temp_file(suffix=".npz", delete=True) as tmpf:
            self.save(tmpf.name)
            result = get_valid_evaluation(tmpf.name,
                                          beam_size=beam_size,
                                          n_jobs=n_jobs,
                                          metric=metric,
                                          f_valid_out=f_valid_out)

        # Return every available metric back
        return result

    def info(self):
        """Reimplement to show model specific information before training."""
        pass

    ##########################################################
    # For all the abstract methods below, you can take a look
    # at attention.py to understand how they are implemented.
    # Remember that you NEED to implement these methods in your
    # own model.
    ##########################################################

    @abstractmethod
    def load_data(self):
        """Load and prepare your training and validation data."""
        pass

    @abstractmethod
    def init_params(self):
        """Initialize the weights and biases of your network."""
        pass

    @abstractmethod
    def build(self):
        """Build the computational graph of your network."""
        pass

    @abstractmethod
    def build_sampler(self, **kwargs):
        """Build f_init() and f_next() for beam-search."""
        pass
