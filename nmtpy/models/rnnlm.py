# -*- coding: utf-8 -*-
from collections import OrderedDict
import numpy as np

import theano
import theano.tensor as tensor
from ..layers import tanh, get_new_layer
from ..defaults import INT, FLOAT
from ..nmtutils import load_dictionary, norm_weight
from ..iterators.text import TextIterator

from .basemodel import BaseModel

class Model(BaseModel):
    def __init__(self, seed, logger, **kwargs):
        # Call parent's init first
        super(Model, self).__init__(**kwargs)

        # Load dictionaries
        dicts = kwargs['dicts']

        # Let's default to GRU
        self.rnn_type = kwargs.get('rnn_type', 'gru')

        self.src_dict, src_idict = load_dictionary(dicts['src'])
        self.n_words = min(self.n_words, len(self.src_dict)) \
                if self.n_words > 0 else len(self.src_dict)

        self.set_options(self.__dict__)
        self.src_idict = src_idict
        self.set_trng(seed)
        self.set_dropout(False)
        self.logger = logger

    def load_valid_data(self):
        self.valid_iterator = TextIterator(
                                batch_size=1,
                                mask=True,
                                shuffle_mode=None,
                                file=self.data['valid_src'],
                                dict=self.src_dict,
                                n_words=self.n_words,
                                name='y') # This is important for the loss to be correctly normalized!
        self.valid_iterator.read()

    def load_data(self):
        self.train_iterator = TextIterator(
                                batch_size=self.batch_size,
                                mask=True,
                                shuffle_mode=None, # or simple or trglen, not tested in rnnlm.
                                file=self.data['train_src'],
                                dict=self.src_dict,
                                n_words=self.n_words)

        self.train_iterator.read()
        self.load_valid_data()

    def init_params(self):
        params = OrderedDict()

        # encoder: ff tanh
        #########
        # Forward encoder initializer
        #params = get_new_layer(self.enc_type)[0](params, prefix='encoder', nin=self.in_emb_dim, nout=self.rnn_dim)
        # embedding weights for encoder
        params['W_in_emb'] = norm_weight(self.n_words, self.in_emb_dim)

        # init_state, init_cell
        #params = get_new_layer('ff')[0](params, prefix='ff_state', nin=self.in_emb_dim, nout=self.rnn_dim)

        # recurrent layer: in_emb_dim to rnn_dim
        params = get_new_layer(self.rnn_type)[0](params, prefix='recurrent', nin=self.in_emb_dim, dim=self.rnn_dim)

        # generate target embedding
        params = get_new_layer('ff')[0](params, prefix='ff_logit_rnn'   , nin=self.rnn_dim, nout=self.out_emb_dim, ortho=False)
        # output to input: out_emb_dim -> out_emb_dim
        params = get_new_layer('ff')[0](params, prefix='ff_logit_prev'  , nin=self.out_emb_dim, nout=self.out_emb_dim, ortho=False)
        # prepare softmax: out_emb_dim -> n_words
        params = get_new_layer('ff')[0](params, prefix='ff_logit'       , nin=self.out_emb_dim, nout=self.n_words)

        self.initial_params = params

    def build(self):
        # description string: #words x #samples
        x = tensor.matrix('x', dtype=INT)
        x_mask = tensor.matrix('x_mask', dtype=FLOAT)

        # Store tensors
        self.inputs = OrderedDict()
        self.inputs['x']        = x         # Source words
        self.inputs['x_mask']   = x_mask    # Source mask

        n_timesteps = x.shape[0]
        n_samples = x.shape[1]

        # input word embedding
        emb = self.tparams['W_in_emb'][x.flatten()]
        emb = emb.reshape([n_timesteps, n_samples, self.in_emb_dim])
        #proj = get_new_layer(self.enc_type)[1](self.tparams, emb, prefix='encoder', mask=x_mask)
        # prepare outputs
        emb_shifted = tensor.zeros_like(emb)
        emb_shifted = tensor.set_subtensor(emb_shifted[1:], emb[:-1])
        emb = emb_shifted

        # pass through gru layer, recurrence here
        proj = get_new_layer(self.rnn_type)[1](self.tparams, emb,
                                                prefix='recurrent', mask=x_mask)

        proj_h = proj[0]

        # compute word probabilities
        # internal state of RNN
        logit_rnn = get_new_layer('ff')[1](self.tparams, proj_h, prefix='ff_logit_rnn', activ='linear')
        # previous output word embedding
        logit_prev = get_new_layer('ff')[1](self.tparams, emb, prefix='ff_logit_prev', activ='linear')
        logit = tanh(logit_rnn + logit_prev)


        logit = get_new_layer('ff')[1](self.tparams, logit, prefix='ff_logit', activ='linear')
        logit_shp = logit.shape

        # Apply logsoftmax (stable version)
        log_probs = -tensor.nnet.logsoftmax(logit.reshape([logit_shp[0]*logit_shp[1], logit_shp[2]]))

        # cost
        x_flat = x.flatten()
        x_flat_idx = tensor.arange(x_flat.shape[0]) * self.n_words + x_flat

        cost = log_probs.flatten()[x_flat_idx]
        cost = cost.reshape([x.shape[0], x.shape[1]])
        cost = (cost * x_mask)

        #f_log_probs_detailled return the log probs array correponding to each word log probs
        self.f_log_probs_detailled = theano.function(list(self.inputs.values()), cost)
        cost = (cost * x_mask).sum(0)

        #f_log_probs return the sum of the sentence log probs
        self.f_log_probs = theano.function(list(self.inputs.values()), cost)

        return cost.mean()

    def val_loss(self, sentence=None):
         probs = []
         if sentence is None:
             #Training validation
             for data in self.valid_iterator:
                 norm = data['y_mask'].sum(0)
                 log_probs = sum(self.f_log_probs(*list(data.values()))) / norm
                 probs.extend(log_probs)
             return np.array(probs).mean()
         else:
             #lm testing one sentence at the time
             norm = sentence['y_mask'].sum(0)
             log_probs = self.f_log_probs_detailled(*list(sentence.values()))
             probs.extend(log_probs)
             return np.array(probs), norm

    def build_sampler(self, **kwargs):
        # x: 1 x 1
        y = tensor.vector('y_sampler', dtype=INT)
        init_state = tensor.matrix('init_state', dtype=FLOAT)

        # if it's the first word, emb should be all zero
        emb = tensor.switch(y[:, None] < 0,
                            tensor.alloc(0., 1, self.tparams['W_in_emb'].shape[1]),
                            self.tparams['W_in_emb'][y])

        # apply one step of gru layer
        proj = get_new_layer(self.rnn_type)[1](self.tparams, emb,
                                                prefix='recurrent',
                                                mask=None)
        next_state = proj[0]

        # compute the output probability dist and sample
        logit_rnn = get_new_layer('ff')[1](self.tparams, next_state, prefix='ff_logit_rnn', activ='linear')
        logit_prev = get_new_layer('ff')[1](self.tparams, emb, prefix='ff_logit_prev', activ='linear')
        logit = tensor.tanh(logit_rnn+logit_prev)
        logit = get_new_layer('ff')[1](self.tparams, logit, prefix='ff_logit', activ='linear')
        logit_shp = logit.shape

        # Apply logsoftmax (stable version)
        next_log_probs = -tensor.nnet.logsoftmax(logit.reshape([logit_shp[0]*logit_shp[1], logit_shp[2]]))

        # Sample from the softmax distribution
        next_probs = tensor.exp(next_log_probs)
        next_word = self.trng.multinomial(pvals=next_probs).argmax(1)

        # next word probability
        inps = [y, init_state]
        outs = [next_log_probs, next_word, next_state]
        self.f_next = theano.function(inps, outs, name='f_next')

    def gen_sample(tparams, f_next, options, trng=None, maxlen=30, argmax=False):
        sample = []
        sample_score = 0

        # initial token is indicated by a -1 and initial state is zero
        next_w = -1 * np.ones((1,)).astype(INT)
        next_state = np.zeros((1, options['dim'])).astype(FLOAT)

        for ii in range(maxlen):
            inps = [next_w, next_state]
            ret = f_next(*inps)
            next_p, next_w, next_state = ret[0], ret[1], ret[2]

            if argmax:
                nw = next_p[0].argmax()
            else:
                nw = next_w[0]
            sample.append(nw)
            sample_score += next_p[0, nw]
            if nw == 0:
                break

        # FIXME: This is broken
        return sample, sample_score, sample_scores, curr_loss, maxlen
