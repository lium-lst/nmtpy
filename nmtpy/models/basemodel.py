# -*- coding: utf-8 -*-
import importlib

from collections import OrderedDict

from abc import ABCMeta, abstractmethod

import theano
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import numpy as np
from ..nmtutils import unzip, get_param_dict
from ..sysutils import readable_size, get_temp_file, get_valid_evaluation
from ..defaults import INT, FLOAT

#######################################
## For debugging function input outputs
def inspect_inputs(i, node, fn):
    print('>> Inputs: ', i, node, [input[0] for input in fn.inputs])

def inspect_outputs(i, node, fn):
    print('>> Outputs: ', i, node, [input[0] for input in fn.outputs])
#######################################

class BaseModel(object, metaclass=ABCMeta):
    def __init__(self, **kwargs):
        # Merge incoming parameters
        self.__dict__.update(kwargs)

        # Will be set when set_dropout is first called
        self.use_dropout    = None

        # Input tensor lists
        self.inputs         = None

        # Theano variables
        self.f_log_probs    = None
        self.f_init         = None
        self.f_next         = None

        self.initial_params = None
        self.tparams        = None

        # Iterators
        self.train_iterator = None
        self.valid_iterator = None

        # A theano shared variable for lrate annealing
        self.learning_rate  = None

    @staticmethod
    def beam_search(inputs, f_inits, f_nexts, beam_size=12, maxlen=100, suppress_unks=False, **kwargs):
        # Override this from your classes
        pass

    def set_options(self, optdict):
        """Filter out None's and save option dict."""
        self.options = OrderedDict([(k,v) for k,v in optdict.items() if v is not None])

    def set_trng(self, seed):
        """Set the seed for Theano RNG."""
        self.trng = RandomStreams(seed)

    def set_dropout(self, val):
        """Set dropout indicator for activation scaling if dropout is available through configuration."""
        if self.use_dropout is None:
            self.use_dropout = theano.shared(np.float64(0.).astype(FLOAT))
        else:
            self.use_dropout.set_value(float(val))

    def update_lrate(self, lrate):
        """Update learning rate."""
        # Update model's value
        self.lrate = lrate
        # Update shared variable used withing the optimizer
        self.learning_rate.set_value(self.lrate)

    def get_nb_params(self):
        """Return the number of parameters of the model."""
        return readable_size(sum([p.size for p in self.initial_params.values()]))

    def save(self, fname):
        """Save model parameters as .npz."""
        if self.tparams is not None:
            np.savez(fname, tparams=unzip(self.tparams), opts=self.options)
        else:
            np.savez(fname, opts=self.options)

    def load(self, fname):
        """Restore .npz checkpoint file into model."""
        self.tparams = OrderedDict()

        params = get_param_dict(fname)
        for k,v in params.items():
            self.tparams[k] = theano.shared(v, name=k)

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

    def build_optimizer(self, cost, regcost, clip_c, dont_update=None, debug=False):
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

        # Load optimizer
        opt = importlib.import_module("nmtpy.optimizers").__dict__[self.optimizer]

        # Create theano shared variable for learning rate
        # self.lrate comes from **kwargs / nmt-train params
        self.learning_rate = theano.shared(np.float64(self.lrate).astype(FLOAT), name='lrate')

        # Get updates
        updates = opt(tparams, grads, self.inputs.values(), final_cost, lr0=self.learning_rate)

        # Compile forward/backward function
        if debug:
            self.train_batch = theano.function(list(self.inputs.values()), norm_cost, updates=updates,
                                               mode=theano.compile.MonitorMode(
                                                   pre_func=inspect_inputs,
                                                   post_func=inspect_outputs))
        else:
            self.train_batch = theano.function(list(self.inputs.values()), norm_cost, updates=updates)

    def run_beam_search(self, beam_size=12, n_jobs=8, metric='bleu', mode='beamsearch', valid_mode='single', f_valid_out=None):
        """Save model under /tmp for passing it to nmt-translate."""
        # Save model temporarily
        with get_temp_file(suffix=".npz", delete=True) as tmpf:
            self.save(tmpf.name)
            result = get_valid_evaluation(tmpf.name,
                                          beam_size=beam_size,
                                          n_jobs=n_jobs,
                                          mode=mode,
                                          metric=metric,
                                          valid_mode=valid_mode,
                                          f_valid_out=f_valid_out)

        # Return every available metric back
        return result

    def gen_sample(self, input_dict, maxlen=100, argmax=False):
        """Generate samples, do greedy (argmax) decoding or forced decoding."""
        # A method that samples or takes the max proba's or
        # does a forced decoding depending on the parameters.
        final_sample = []
        final_score = 0

        target = None
        if "y_true" in input_dict:
            # We're doing forced decoding
            target = input_dict.pop("y_true")
            maxlen = len(target)

        inputs = list(input_dict.values())

        next_state, ctx0 = self.f_init(*inputs)

        # Beginning-of-sentence indicator is -1
        next_word = np.array([-1], dtype=INT)

        for ii in range(maxlen):
            # Get next states
            next_log_p, next_word, next_state = self.f_next(*[next_word, ctx0, next_state])

            if target is not None:
                nw = int(target[ii])

            elif argmax:
                # argmax() works the same for both probas and log_probas
                nw = next_log_p[0].argmax()

            else:
                # Multinomial sampling
                nw = next_word[0]

            # 0: <eos>
            if nw == 0:
                break

            # Add the word idx
            final_sample.append(nw)
            final_score -= next_log_p[0, nw]

        final_sample = [final_sample]
        final_score = np.array(final_score)

        return final_sample, final_score

    def generate_samples(self, batch_dict, n_samples):
        # Silently fail if generate_samples is not reimplemented
        # in child classes
        return None

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
    def build_sampler(self):
        """Similar to build() but works sequentially for beam-search or sampling."""
        pass
