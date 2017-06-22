# -*- coding: utf-8 -*-

# Python
from collections import OrderedDict, defaultdict
import tempfile
import os

# 3rd party
import numpy as np

import theano
import theano.tensor as tensor

# Ours
from ..layers import *
from ..defaults import INT, FLOAT
from ..nmtutils import *
from ..iterators.text import TextIterator
from ..iterators.bitext import BiTextIterator
from ..iterators.factors import FactorsIterator
from .basemodel import BaseModel
from ..sysutils import readable_size, get_temp_file, get_valid_evaluation
from .basefnmt import Model as AttentionFnmt

class Model(AttentionFnmt):
    ###################################################################
    # The following methods can be redefined in child models inheriting
    # from this basic Attention model.
    ###################################################################
    def init_params(self):
        params = OrderedDict()

        # embedding weights for encoder and decoder
        params['Wemb_enc'] = norm_weight(self.n_words_src, self.embedding_dim, scale=self.weight_init)
        params['Wemb_dec_lem'] = norm_weight(self.n_words_trg1, self.embedding_dim, scale=self.weight_init)
        params['Wemb_dec_fact'] = norm_weight(self.n_words_trg2, self.embedding_dim, scale=self.weight_init)

        ############################
        # encoder: bidirectional RNN
        ############################
        # Forward encoder
        params = get_new_layer(self.enc_type)[0](params, prefix='encoder', nin=self.embedding_dim, dim=self.rnn_dim, scale=self.weight_init, layernorm=self.lnorm)
        # Backwards encoder
        params = get_new_layer(self.enc_type)[0](params, prefix='encoder_r', nin=self.embedding_dim, dim=self.rnn_dim, scale=self.weight_init, layernorm=self.lnorm)

        # How many additional encoder layers to stack?
        for i in range(1, self.n_enc_layers):
            params = get_new_layer(self.enc_type)[0](params, prefix='deep_encoder_%d' % i,
                                                     nin=self.ctx_dim, dim=self.ctx_dim,
                                                     scale=self.weight_init, layernorm=self.lnorm)

        ############################
        # How do we initialize CGRU?
        ############################
        if self.init_cgru == 'text':
            # init_state computation from mean textual context
            params = get_new_layer('ff')[0](params, prefix='ff_state', nin=self.ctx_dim, nout=self.rnn_dim, scale=self.weight_init)

        #########
        # decoder
        #########
        params = get_new_layer('gru_cond')[0](params, prefix='decoder', nin=2*self.embedding_dim, dim=self.rnn_dim, dimctx=self.ctx_dim, scale=self.weight_init, layernorm=False)

        ########
        # fusion
        ########
        params = get_new_layer('ff')[0](params, prefix='ff_logit_gru'  , nin=self.rnn_dim       , nout=self.embedding_dim, scale=self.weight_init, ortho=False)
        params = get_new_layer('ff')[0](params, prefix='ff_logit_lem' ,  nin=self.embedding_dim , nout=self.embedding_dim, scale=self.weight_init, ortho=False)
        params = get_new_layer('ff')[0](params, prefix='ff_logit_fact' , nin=self.embedding_dim , nout=self.embedding_dim, scale=self.weight_init, ortho=False)
        params = get_new_layer('ff')[0](params, prefix='ff_logit_ctx'  , nin=self.ctx_dim       , nout=self.embedding_dim, scale=self.weight_init, ortho=False)
        if self.tied_trg_emb is False:
            params = get_new_layer('ff')[0](params, prefix='ff_logit_trg'  , nin=self.embedding_dim , nout=self.n_words_trg1, scale=self.weight_init)
            params = get_new_layer('ff')[0](params, prefix='ff_logit_trgmult'  , nin=self.embedding_dim , nout=self.n_words_trg2, scale=self.weight_init)

        self.initial_params = params

    def build(self):
        # description string: #words x #samples
        x = tensor.matrix('x', dtype=INT)
        x_mask = tensor.matrix('x_mask', dtype=FLOAT)
        y1 = tensor.matrix('y1', dtype=INT)
        y1_mask = tensor.matrix('y1_mask', dtype=FLOAT)
        y2 = tensor.matrix('y2', dtype=INT)
        y2_mask = tensor.matrix('y2_mask', dtype=FLOAT)

        self.inputs = OrderedDict()
        self.inputs['x'] = x
        self.inputs['x_mask'] = x_mask
        self.inputs['y1'] = y1
        self.inputs['y2'] = y2
        self.inputs['y1_mask'] = y1_mask
        self.inputs['y2_mask'] = y2_mask

        # for the backward rnn, we just need to invert x and x_mask
        xr = x[::-1]
        xr_mask = x_mask[::-1]

        n_timesteps = x.shape[0]
        n_timesteps_trg = y1.shape[0]
        n_timesteps_trgmult = y2.shape[0]
        n_samples = x.shape[1]

        # word embedding for forward rnn (source)
        emb = dropout(self.tparams['Wemb_enc'][x.flatten()],
                      self.trng, self.emb_dropout, self.use_dropout)
        emb = emb.reshape([n_timesteps, n_samples, self.embedding_dim])
        proj = get_new_layer(self.enc_type)[1](self.tparams, emb, prefix='encoder', mask=x_mask, layernorm=self.lnorm)

        # word embedding for backward rnn (source)
        embr = dropout(self.tparams['Wemb_enc'][xr.flatten()],
                       self.trng, self.emb_dropout, self.use_dropout)
        embr = embr.reshape([n_timesteps, n_samples, self.embedding_dim])
        projr = get_new_layer(self.enc_type)[1](self.tparams, embr, prefix='encoder_r', mask=xr_mask, layernorm=self.lnorm)

        # context will be the concatenation of forward and backward rnns
        ctx = [tensor.concatenate([proj[0], projr[0][::-1]], axis=proj[0].ndim-1)]

        for i in range(1, self.n_enc_layers):
            ctx = get_new_layer(self.enc_type)[1](self.tparams, ctx[0],
                                                  prefix='deepencoder_%d' % i,
                                                  mask=x_mask, layernorm=self.lnorm)

        # Apply dropout
        ctx = dropout(ctx[0], self.trng, self.ctx_dropout, self.use_dropout)

        if self.init_cgru == 'text':
            # mean of the context (across time) will be used to initialize decoder rnn
            ctx_mean   = (ctx * x_mask[:, :, None]).sum(0) / x_mask.sum(0)[:, None]
            init_state = get_new_layer('ff')[1](self.tparams, ctx_mean, prefix='ff_state', activ='tanh')
        else:
            # Assume zero-initialized decoder
            init_state = tensor.alloc(0., n_samples, self.rnn_dim)

        # word embedding (target), we will shift the target sequence one time step
        # to the right. This is done because of the bi-gram connections in the
        # readout and decoder rnn. The first target will be all zeros and we will
        emb_lem = self.tparams['Wemb_dec_lem'][y1.flatten()]
        emb_lem = emb_lem.reshape([n_timesteps_trg, n_samples, self.embedding_dim])
        emb_lem_shifted = tensor.zeros_like(emb_lem)
        emb_lem_shifted = tensor.set_subtensor(emb_lem_shifted[1:], emb_lem[:-1])
        emb_lem = emb_lem_shifted

        emb_fact = self.tparams['Wemb_dec_fact'][y2.flatten()]
        emb_fact = emb_fact.reshape([n_timesteps_trgmult, n_samples, self.embedding_dim])
        emb_fact_shifted = tensor.zeros_like(emb_fact)
        emb_fact_shifted = tensor.set_subtensor(emb_fact_shifted[1:], emb_fact[:-1])
        emb_fact = emb_fact_shifted
    
        # Concat the 2 embeddings
        emb_prev = tensor.concatenate([emb_lem, emb_fact], axis=2)
    
        # decoder - pass through the decoder conditional gru with attention
        proj = get_new_layer('gru_cond')[1](self.tparams, emb_prev,
                                            prefix='decoder',
                                            mask=y1_mask, context=ctx,
                                            context_mask=x_mask,
                                            one_step=False,
                                            init_state=init_state, layernorm=False)
        # hidden states of the decoder gru
        proj_h = proj[0]

        # weighted averages of context, generated by attention module
        ctxs = proj[1]

        # weights (alignment matrix)
        self.alphas = proj[2]

        # compute word probabilities
        logit_gru  = get_new_layer('ff')[1](self.tparams, proj_h, prefix='ff_logit_gru', activ='linear')
        logit_ctx  = get_new_layer('ff')[1](self.tparams, ctxs, prefix='ff_logit_ctx', activ='linear')
        logit_lem = get_new_layer('ff')[1](self.tparams, emb_lem, prefix='ff_logit_lem', activ='linear')
        logit_fact = get_new_layer('ff')[1](self.tparams, emb_fact, prefix='ff_logit_fact', activ='linear')

        logit1 = dropout(tanh(logit_gru + logit_lem + logit_ctx), self.trng, self.out_dropout, self.use_dropout)
        logit2 = dropout(tanh(logit_gru + logit_fact + logit_ctx), self.trng, self.out_dropout, self.use_dropout)

        if self.tied_trg_emb is False:
            logit_trg = get_new_layer('ff')[1](self.tparams, logit1, prefix='ff_logit_trg', activ='linear')
            logit_trgmult = get_new_layer('ff')[1](self.tparams, logit2, prefix='ff_logit_trgmult', activ='linear')
        
        else:
            logit_trg = tensor.dot(logit1, self.tparams['Wemb_dec_lem'].T)
            logit_trgmult = tensor.dot(logit2, self.tparams['Wemb_dec_fact'].T)

        logit_trg_shp = logit_trg.shape
        logit_trgmult_shp = logit_trgmult.shape

        # Apply logsoftmax (stable version)
        log_trg_probs = -tensor.nnet.logsoftmax(logit_trg.reshape([logit_trg_shp[0]*logit_trg_shp[1], logit_trg_shp[2]]))
        log_trgmult_probs = -tensor.nnet.logsoftmax(logit_trgmult.reshape([logit_trgmult_shp[0]*logit_trgmult_shp[1], logit_trgmult_shp[2]]))

        # cost
        y1_flat = y1.flatten()
        y2_flat = y2.flatten()
        y1_flat_idx = tensor.arange(y1_flat.shape[0]) * self.n_words_trg1 + y1_flat
        y2_flat_idx = tensor.arange(y2_flat.shape[0]) * self.n_words_trg2 + y2_flat

        cost_trg = log_trg_probs.flatten()[y1_flat_idx]
        cost_trg = cost_trg.reshape([n_timesteps_trg, n_samples])
        cost_trg = (cost_trg * y1_mask).sum(0)

        cost_trgmult = log_trgmult_probs.flatten()[y2_flat_idx]
        cost_trgmult = cost_trgmult.reshape([n_timesteps_trgmult, n_samples])
        cost_trgmult = (cost_trgmult * y2_mask).sum(0)

        cost = cost_trg + cost_trgmult
        self.f_log_probs = theano.function(list(self.inputs.values()), cost)

        # For alpha regularization

        return cost

    def build_sampler(self):
        x           = tensor.matrix('x', dtype=INT)
        xr          = x[::-1]
        n_timesteps = x.shape[0]
        n_samples   = x.shape[1]

        # word embedding (source), forward and backward
        emb = self.tparams['Wemb_enc'][x.flatten()]
        emb = emb.reshape([n_timesteps, n_samples, self.embedding_dim])

        embr = self.tparams['Wemb_enc'][xr.flatten()]
        embr = embr.reshape([n_timesteps, n_samples, self.embedding_dim])

        # encoder
        proj = get_new_layer(self.enc_type)[1](self.tparams, emb, prefix='encoder', layernorm=self.lnorm)
        projr = get_new_layer(self.enc_type)[1](self.tparams, embr, prefix='encoder_r', layernorm=self.lnorm)

        # concatenate forward and backward rnn hidden states
        ctx = [tensor.concatenate([proj[0], projr[0][::-1]], axis=proj[0].ndim-1)]

        for i in range(1, self.n_enc_layers):
            ctx = get_new_layer(self.enc_type)[1](self.tparams, ctx[0],
                                                  prefix='deepencoder_%d' % i,
                                                  layernorm=self.lnorm)

        ctx = ctx[0]

        if self.init_cgru == 'text' and 'ff_state_W' in self.tparams:
            # get the input for decoder rnn initializer mlp
            ctx_mean = ctx.mean(0)
            init_state = get_new_layer('ff')[1](self.tparams, ctx_mean, prefix='ff_state', activ='tanh')
        else:
            # assume zero-initialized decoder
            init_state = tensor.alloc(0., n_samples, self.rnn_dim)

        outs = [init_state, ctx]
        self.f_init = theano.function([x], outs, name='f_init')

        # x: 1 x 1
        y1 = tensor.vector('y1_sampler', dtype=INT)
        y2 = tensor.vector('y2_sampler', dtype=INT)
        init_state = tensor.matrix('init_state', dtype=FLOAT)

        # if it's the first word, emb should be all zero and it is indicated by -1
        emb_lem = tensor.switch(y1[:, None] < 0,
                            tensor.alloc(0., 1, self.tparams['Wemb_dec_lem'].shape[1]),
                            self.tparams['Wemb_dec_lem'][y1])
        emb_fact = tensor.switch(y2[:, None] < 0,
                            tensor.alloc(0., 1, self.tparams['Wemb_dec_fact'].shape[1]),
                            self.tparams['Wemb_dec_fact'][y2])
        
        # Concat the 2 embeddings
        emb_prev = tensor.concatenate([emb_lem,emb_fact], axis=1)

        # apply one step of conditional gru with attention
        # get the next hidden states
        # get the weighted averages of contexts for this target word y
        r = get_new_layer('gru_cond')[1](self.tparams, emb_prev,
                                         prefix='decoder',
                                         mask=None, context=ctx,
                                         one_step=True,
                                         init_state=init_state, layernorm=False)

        next_state = r[0]
        ctxs = r[1]
        alphas = r[2]

        logit_lem  = get_new_layer('ff')[1](self.tparams, emb_lem,      prefix='ff_logit_lem' ,activ='linear')
        logit_fact = get_new_layer('ff')[1](self.tparams, emb_fact,     prefix='ff_logit_fact',activ='linear')
        logit_ctx  = get_new_layer('ff')[1](self.tparams, ctxs,         prefix='ff_logit_ctx', activ='linear')
        logit_gru  = get_new_layer('ff')[1](self.tparams, next_state,   prefix='ff_logit_gru', activ='linear')

        logit1 = tanh(logit_gru + logit_lem + logit_ctx)
        logit2 = tanh(logit_gru + logit_fact + logit_ctx)

        if self.tied_trg_emb is False:
            logit = get_new_layer('ff')[1](self.tparams, logit1, prefix='ff_logit', activ='linear')
            logit_trgmult = get_new_layer('ff')[1](self.tparams, logit2, prefix='ff_logit_trgmult', activ='linear')
        else:
            logit_trg = tensor.dot(logit1, self.tparams['Wemb_dec_lem'].T)
            logit_trgmult = tensor.dot(logit2, self.tparams['Wemb_dec_fact'].T)

        # compute the logsoftmax
        next_log_probs_trg = tensor.nnet.logsoftmax(logit_trg)
        next_log_probs_trgmult = tensor.nnet.logsoftmax(logit_trgmult)

        # Sample from the softmax distribution
        next_probs_trg = tensor.exp(next_log_probs_trg)
        next_probs_trgmult = tensor.exp(next_log_probs_trgmult)
        next_word_trg = self.trng.multinomial(pvals=next_probs_trg).argmax(1)
        next_word_trgmult = self.trng.multinomial(pvals=next_probs_trgmult).argmax(1)

        # NOTE: We never use sampling and it incurs performance penalty
        # let's disable it for now
        #next_word = self.trng.multinomial(pvals=next_probs).argmax(1)

        # compile a function to do the whole thing above
        # next hidden state to be used
        inputs = [y1, y2, init_state, ctx]
        outs = [next_log_probs_trg, next_log_probs_trgmult, next_state, alphas]

        self.f_next = theano.function(inputs, outs, name='f_next')
