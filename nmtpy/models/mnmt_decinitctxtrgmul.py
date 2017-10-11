# -*- coding: utf-8 -*-
from collections import OrderedDict

import theano
import theano.tensor as T

import numpy as np

# Ours
from ..layers import dropout, tanh, get_new_layer
from ..defaults import INT, FLOAT
from ..nmtutils import norm_weight
from ..iterators.mnmt import MNMTIterator

from .attention import Model as Attention

############################################################
# Attentive NMT + Decoder initialized with FC image features
############################################################

class Model(Attention):
    def __init__(self, seed, logger, **kwargs):
        # Call parent's init first
        super(Model, self).__init__(seed, logger, **kwargs)

    def load_data(self):
        # Load training data
        self.train_iterator = MNMTIterator(
                batch_size=self.batch_size,
                logger=self._logger,
                pklfile=self.data['train_src'],
                imgfile=self.data['train_img'],
                trgdict=self.trg_dict,
                srcdict=self.src_dict,
                n_words_trg=self.n_words_trg, n_words_src=self.n_words_src)
        self.train_iterator.read()
        self.load_valid_data()

    def load_valid_data(self, from_translate=False, data_mode='single'):
        # Load validation data
        batch_size = 1 if from_translate else 64
        if from_translate:
            self.valid_ref_files = self.data['valid_trg']
            if isinstance(self.valid_ref_files, str):
                self.valid_ref_files = list([self.valid_ref_files])

            self.valid_iterator = MNMTIterator(
                    batch_size=batch_size,
                    mask=False,
                    pklfile=self.data['valid_src'],
                    imgfile=self.data['valid_img'],
                    srcdict=self.src_dict, n_words_src=self.n_words_src)
        else:
            # Just for loss computation
            self.valid_iterator = MNMTIterator(
                    batch_size=self.batch_size,
                    pklfile=self.data['valid_src'],
                    imgfile=self.data['valid_img'],
                    trgdict=self.trg_dict, srcdict=self.src_dict,
                    n_words_trg=self.n_words_trg, n_words_src=self.n_words_src)

        self.valid_iterator.read()

    def init_params(self):
        params = OrderedDict()

        # embedding weights for encoder and decoder
        params[self.src_emb_name] = norm_weight(self.n_words_src, self.embedding_dim, scale=self.weight_init)
        if self.tied_emb != '3way':
            params[self.trg_emb_name] = norm_weight(self.n_words_trg, self.embedding_dim, scale=self.weight_init)

        ############################
        # encoder: bidirectional RNN
        ############################
        # Forward encoder
        params = get_new_layer(self.enc_type)[0](params, prefix='encoder', nin=self.embedding_dim,
                                                 dim=self.rnn_dim, scale=self.weight_init, layernorm=self.layer_norm)
        # Backwards encoder
        params = get_new_layer(self.enc_type)[0](params, prefix='encoder_r', nin=self.embedding_dim,
                                                 dim=self.rnn_dim, scale=self.weight_init, layernorm=self.layer_norm)

        # How many additional encoder layers to stack?
        for i in range(1, self.n_enc_layers):
            params = get_new_layer(self.enc_type)[0](params, prefix='deep_encoder_%d' % i,
                                                     nin=self.ctx_dim, dim=self.ctx_dim,
                                                     scale=self.weight_init, layernorm=self.layer_norm)


        params = get_new_layer('ff')[0](params, prefix='ff_img_init', nin=self.img_dim,
                                        nout=self.rnn_dim, scale=self.weight_init)
        params = get_new_layer('ff')[0](params, prefix='ff_img_ymul', nin=self.img_dim,
                                        nout=self.embedding_dim, scale=self.weight_init)
        params = get_new_layer('ff')[0](params, prefix='ff_img_xmul', nin=self.img_dim,
                                        nout=self.ctx_dim, scale=self.weight_init)

        #########
        # decoder
        #########
        params = get_new_layer('gru_cond')[0](params, prefix='decoder', nin=self.embedding_dim,
                                              dim=self.rnn_dim, dimctx=self.ctx_dim,
                                              scale=self.weight_init, layernorm=False)

        ########
        # fusion
        ########
        params = get_new_layer('ff')[0](params, prefix='ff_logit_gru'  , nin=self.rnn_dim       , nout=self.embedding_dim, scale=self.weight_init, ortho=False)
        params = get_new_layer('ff')[0](params, prefix='ff_logit_prev' , nin=self.embedding_dim , nout=self.embedding_dim, scale=self.weight_init, ortho=False)
        params = get_new_layer('ff')[0](params, prefix='ff_logit_ctx'  , nin=self.ctx_dim       , nout=self.embedding_dim, scale=self.weight_init, ortho=False)
        if self.tied_emb is False:
            params = get_new_layer('ff')[0](params, prefix='ff_logit'  , nin=self.embedding_dim , nout=self.n_words_trg, scale=self.weight_init)

        self.initial_params = params

    def build(self):
        x                       = T.matrix('x', dtype=INT)
        x_mask                  = T.matrix('x_mask', dtype=FLOAT)
        x_img                   = T.matrix('x_img', dtype=FLOAT)
        y                       = T.matrix('y', dtype=INT)
        y_mask                  = T.matrix('y_mask', dtype=FLOAT)

        self.inputs             = OrderedDict()
        self.inputs['x']        = x
        self.inputs['x_mask']   = x_mask
        self.inputs['x_img']    = x_img     # Image features
        self.inputs['y']        = y
        self.inputs['y_mask']   = y_mask

        # for the backward rnn, we just need to invert x and x_mask
        xr      = x[::-1]
        xr_mask = x_mask[::-1]

        n_timesteps, n_samples  = x.shape
        n_timesteps_trg         = y.shape[0]

        # Transform pool5 features for target embedding multiplication
        img_ymul = get_new_layer('ff')[1](self.tparams, x_img, prefix='ff_img_ymul', activ='tanh')
        # Transform pool5 features for source context multiplication
        img_xmul = get_new_layer('ff')[1](self.tparams, x_img, prefix='ff_img_xmul', activ='tanh')
        # Decoder initialized with pool5 features
        init_state = get_new_layer('ff')[1](self.tparams, x_img, prefix='ff_img_init', activ='tanh')

        # word embedding for forward rnn (source)
        emb = dropout(self.tparams[self.src_emb_name][x.flatten()],
                      self.trng, self.emb_dropout, self.use_dropout)
        emb = emb.reshape([n_timesteps, n_samples, self.embedding_dim])
        proj = get_new_layer(self.enc_type)[1](self.tparams, emb, prefix='encoder', mask=x_mask, layernorm=self.layer_norm)

        # word embedding for backward rnn (source)
        embr = dropout(self.tparams[self.src_emb_name][xr.flatten()],
                       self.trng, self.emb_dropout, self.use_dropout)
        embr = embr.reshape([n_timesteps, n_samples, self.embedding_dim])
        projr = get_new_layer(self.enc_type)[1](self.tparams, embr, prefix='encoder_r', mask=xr_mask, layernorm=self.layer_norm)

        # context will be the concatenation of forward and backward rnns
        ctx = [T.concatenate([proj[0], projr[0][::-1]], axis=proj[0].ndim-1)]

        for i in range(1, self.n_enc_layers):
            ctx = get_new_layer(self.enc_type)[1](self.tparams, ctx[0],
                                                  prefix='deepencoder_%d' % i,
                                                  mask=x_mask, layernorm=self.layer_norm)

        # Apply dropout to source context
        ctx = dropout(ctx[0], self.trng, self.ctx_dropout, self.use_dropout)

        # Modulate source context with visual features
        ctx = ctx * img_xmul[None, ...]

        # word embedding (target), we will shift the target sequence one time step
        # to the right. This is done because of the bi-gram connections in the
        # readout and decoder rnn. The first target will be all zeros and we will
        # not condition on the last output.
        emb         = self.tparams[self.trg_emb_name][y.flatten()]
        emb         = emb.reshape([n_timesteps_trg, n_samples, self.embedding_dim])

        # Multiply target embeddings with image features
        emb         = emb * img_ymul[None, ...]
        emb_shifted = T.zeros_like(emb)
        emb_shifted = T.set_subtensor(emb_shifted[1:], emb[:-1])
        emb         = emb_shifted

        # decoder - pass through the decoder conditional gru with attention
        # init_state is zero
        r = get_new_layer('gru_cond')[1](self.tparams, emb,
                                         prefix='decoder',
                                         mask=y_mask, context=ctx,
                                         context_mask=x_mask,
                                         one_step=False,
                                         init_state=init_state, layernorm=False)
        # hidden states of the decoder gru
        # weighted averages of context, generated by attention module
        # weights (alignment matrix)
        next_state, ctxs, alphas = r

        # compute word probabilities
        logit_gru  = get_new_layer('ff')[1](self.tparams, next_state, prefix='ff_logit_gru', activ='linear')
        logit_ctx  = get_new_layer('ff')[1](self.tparams, ctxs, prefix='ff_logit_ctx', activ='linear')
        logit_prev = get_new_layer('ff')[1](self.tparams, emb, prefix='ff_logit_prev', activ='linear')

        logit = dropout(tanh(logit_gru + logit_prev + logit_ctx), self.trng, self.out_dropout, self.use_dropout)

        if self.tied_emb is False:
            logit = get_new_layer('ff')[1](self.tparams, logit, prefix='ff_logit', activ='linear')
        else:
            logit = T.dot(logit, self.tparams[self.trg_emb_name].T)

        logit_shp = logit.shape

        # Apply logsoftmax (stable version)
        log_probs = -T.nnet.logsoftmax(logit.reshape([logit_shp[0]*logit_shp[1], logit_shp[2]]))

        # cost
        y_flat = y.flatten()
        y_flat_idx = T.arange(y_flat.shape[0]) * self.n_words_trg + y_flat

        cost = log_probs.flatten()[y_flat_idx]
        cost = cost.reshape([n_timesteps_trg, n_samples])
        cost = (cost * y_mask).sum(0)

        self.f_log_probs = theano.function(list(self.inputs.values()), cost)

        # Very slowly increasing multiplicative penalty for longer sequences
        #penalty = (0.002*T.arange(n_timesteps_trg)**2) + 1
        #return ((cost * penalty[:, None]) * y_mask).sum(0)

        return cost

    def build_sampler(self, **kwargs):
        x       = T.matrix('x', dtype=INT)
        x_img   = T.matrix('x_img', dtype=FLOAT)
        xr      = x[::-1]

        n_timesteps, n_samples = x.shape

        # Transform pool5 features for target embedding multiplication
        img_ymul = get_new_layer('ff')[1](self.tparams, x_img, prefix='ff_img_ymul', activ='tanh')
        # Transform pool5 features for source context multiplication
        img_xmul = get_new_layer('ff')[1](self.tparams, x_img, prefix='ff_img_xmul', activ='tanh')
        # Decoder initialized with pool5 features
        init_state = get_new_layer('ff')[1](self.tparams, x_img, prefix='ff_img_init', activ='tanh')

        # word embedding (source), forward and backward
        emb = self.tparams[self.src_emb_name][x.flatten()]
        emb = emb.reshape([n_timesteps, n_samples, self.embedding_dim])

        embr = self.tparams[self.src_emb_name][xr.flatten()]
        embr = embr.reshape([n_timesteps, n_samples, self.embedding_dim])

        # encoder
        proj  = get_new_layer(self.enc_type)[1](self.tparams, emb, prefix='encoder', layernorm=self.layer_norm)
        projr = get_new_layer(self.enc_type)[1](self.tparams, embr, prefix='encoder_r', layernorm=self.layer_norm)

        # concatenate forward and backward rnn hidden states
        ctx = [T.concatenate([proj[0], projr[0][::-1]], axis=proj[0].ndim-1)]

        for i in range(1, self.n_enc_layers):
            ctx = get_new_layer(self.enc_type)[1](self.tparams, ctx[0],
                                                  prefix='deepencoder_%d' % i,
                                                  layernorm=self.layer_norm)

        ctx = ctx[0] * img_xmul[None, ...]

        outs = [init_state, ctx, img_ymul]
        self.f_init = theano.function([x, x_img], outs, name='f_init')

        # x: 1 x 1
        y = T.vector('y_sampler', dtype=INT)
        init_state = T.matrix('init_state', dtype=FLOAT)
        img_ymul = T.TensorType(FLOAT, (True, False))('img_ymul')

        # if it's the first word, emb should be all zero and it is indicated by -1
        emb = T.switch(y[:, None] < 0,
                            T.alloc(0., 1, self.tparams[self.trg_emb_name].shape[1]),
                            self.tparams[self.trg_emb_name][y] * img_ymul)        # Multiply emb with img_ymul

        # apply one step of conditional gru with attention
        # get the next hidden states
        # get the weighted averages of contexts for this target word y
        # init_state is zero
        r = get_new_layer('gru_cond')[1](self.tparams, emb,
                                         prefix='decoder',
                                         mask=None, context=ctx,
                                         one_step=True,
                                         init_state=init_state, layernorm=False)

        next_state, ctxs, alphas = r

        logit_prev = get_new_layer('ff')[1](self.tparams, emb,          prefix='ff_logit_prev',activ='linear')
        logit_ctx  = get_new_layer('ff')[1](self.tparams, ctxs,         prefix='ff_logit_ctx', activ='linear')
        logit_gru  = get_new_layer('ff')[1](self.tparams, next_state,   prefix='ff_logit_gru', activ='linear')

        logit = tanh(logit_gru + logit_prev + logit_ctx)

        if self.tied_emb is False:
            logit = get_new_layer('ff')[1](self.tparams, logit, prefix='ff_logit', activ='linear')
        else:
            logit = T.dot(logit, self.tparams[self.trg_emb_name].T)

        # compute the logsoftmax
        next_log_probs = T.nnet.logsoftmax(logit)

        # compile a function to do the whole thing above
        # next hidden state to be used
        inputs = [y, init_state, ctx, img_ymul]

        outs = [next_log_probs, next_state, alphas]
        self.f_next = theano.function(inputs, outs, name='f_next')
