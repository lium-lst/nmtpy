# -*- coding: utf-8 -*-
from collections import OrderedDict

# 3rd party
import numpy as np

import theano
import theano.tensor as tensor

# Ours
from ..layers import dropout, get_new_layer, tanh
from ..defaults import INT, FLOAT
from ..nmtutils import norm_weight
from ..iterators.wmt import WMTIterator

from .attention import Model as Attention

# Base Multimodal (fusion) model without a
# multimodal decoder implementation. Child classes should
# derive from this and set init_gru_decoder and gru_decoder
# to their implementations.

# NOTE: This is 'not' a model on its own, do not set this
# as your model_type in configuration files!

class Model(Attention):
    def __init__(self, seed, logger, **kwargs):
        # Call Attention's __init__
        super(Model, self).__init__(seed, logger, **kwargs)

        # These should be set by child models depending
        # on their specific decoder implementations
        self.init_gru_decoder   = None
        self.gru_decoder        = None

    def info(self):
        self.logger.info('Source vocabulary size: %d', self.n_words_src)
        self.logger.info('Target vocabulary size: %d', self.n_words_trg)
        self.logger.info('%d training samples' % self.train_iterator.n_samples)
        self.logger.info('  %d/%d UNKs in source, %d/%d UNKs in target' % (self.train_iterator.unk_src,
                                                                           self.train_iterator.total_src_words,
                                                                           self.train_iterator.unk_trg,
                                                                           self.train_iterator.total_trg_words))
        self.logger.info('%d validation samples' % self.valid_iterator.n_samples)
        self.logger.info('  %d UNKs in source' % self.valid_iterator.unk_src)
        self.logger.info('dropout (emb,ctx,out): %.2f, %.2f, %.2f' % (self.emb_dropout, self.ctx_dropout, self.out_dropout))

    def load_data(self):
        # Load training data
        self.train_iterator = WMTIterator(
                batch_size=self.batch_size,
                shuffle_mode=self.smode,
                logger=self.logger,
                pklfile=self.data['train_src'],
                imgfile=self.data['train_img'],
                trgdict=self.trg_dict,
                srcdict=self.src_dict,
                n_words_trg=self.n_words_trg, n_words_src=self.n_words_src,
                mode=self.options.get('data_mode', 'pairs'))
        self.train_iterator.read()
        self.load_valid_data()

    def load_valid_data(self, from_translate=False, data_mode='single'):
        # Load validation data
        batch_size = 1 if from_translate else 64
        if from_translate:
            self.valid_ref_files = self.data['valid_trg']
            if isinstance(self.valid_ref_files, str):
                self.valid_ref_files = list([self.valid_ref_files])

            self.valid_iterator = WMTIterator(
                    batch_size=batch_size,
                    mask=False,
                    pklfile=self.data['valid_src'],
                    imgfile=self.data['valid_img'],
                    srcdict=self.src_dict, n_words_src=self.n_words_src,
                    mode=data_mode)
        else:
            # Just for loss computation
            self.valid_iterator = WMTIterator(
                    batch_size=self.batch_size,
                    pklfile=self.data['valid_src'],
                    imgfile=self.data['valid_img'],
                    trgdict=self.trg_dict, srcdict=self.src_dict,
                    n_words_trg=self.n_words_trg, n_words_src=self.n_words_src,
                    mode='single')

        self.valid_iterator.read()

    def init_params(self):
        if self.init_gru_decoder is None:
            raise Exception('Base fusion model should not be instantiated directly.')

        params = OrderedDict()

        # embedding weights for encoder (source language)
        params['Wemb_enc'] = norm_weight(self.n_words_src, self.embedding_dim, scale=self.weight_init)

        # embedding weights for decoder (target language)
        params['Wemb_dec'] = norm_weight(self.n_words_trg, self.embedding_dim, scale=self.weight_init)

        # convfeats (1024) to ctx dim (2000) for image modality
        params = get_new_layer('ff')[0](params, prefix='ff_img_adaptor', nin=self.conv_dim,
                                        nout=self.ctx_dim, scale=self.weight_init)

        #############################################
        # Source sentence encoder: bidirectional GRU
        #############################################
        # Forward and backward encoder parameters
        params = get_new_layer('gru')[0](params, prefix='text_encoder', nin=self.embedding_dim,
                                         dim=self.rnn_dim, scale=self.weight_init, layernorm=self.lnorm)
        params = get_new_layer('gru')[0](params, prefix='text_encoder_r', nin=self.embedding_dim,
                                         dim=self.rnn_dim, scale=self.weight_init, layernorm=self.lnorm)

        ##########
        # Decoder
        ##########
        if self.init_cgru == 'text':
            # init_state computation from mean textual context
            params = get_new_layer('ff')[0](params, prefix='ff_text_state_init', nin=self.ctx_dim,
                                            nout=self.rnn_dim, scale=self.weight_init)
        elif self.init_cgru == 'img':
            # Global average pooling to init the decoder
            params = get_new_layer('ff')[0](params, prefix='ff_img_state_init', nin=self.conv_dim,
                                            nout=self.rnn_dim, scale=self.weight_init)
        elif self.init_cgru == 'textimg':
            # A combination of both modalities
            params = get_new_layer('ff')[0](params, prefix='ff_textimg_state_init', nin=self.ctx_dim+self.conv_dim,
                                            nout=self.rnn_dim, scale=self.weight_init)

        # GRU cond decoder
        params = self.init_gru_decoder(params, prefix='decoder_multi', nin=self.embedding_dim,
                                        dim=self.rnn_dim, dimctx=self.ctx_dim, scale=self.weight_init)

        # readout
        params = get_new_layer('ff')[0](params, prefix='ff_logit_gru', nin=self.rnn_dim,
                                        nout=self.embedding_dim, scale=self.weight_init)
        params = get_new_layer('ff')[0](params, prefix='ff_logit_ctx', nin=self.ctx_dim,
                                        nout=self.embedding_dim, scale=self.weight_init)
        if self.tied_trg_emb is False:
            params = get_new_layer('ff')[0](params, prefix='ff_logit', nin=self.embedding_dim,
                                            nout=self.n_words_trg, scale=self.weight_init)

        # Save initial parameters for debugging purposes
        self.initial_params = params

    def build(self):
        # Source sentences: n_timesteps, n_samples
        x       = tensor.matrix('x', dtype=INT)
        x_mask  = tensor.matrix('x_mask', dtype=FLOAT)

        # Image: 196 (n_annotations) x n_samples x 1024 (conv_dim)
        x_img   = tensor.tensor3('x_img', dtype=FLOAT)

        # Target sentences: n_timesteps, n_samples
        y       = tensor.matrix('y', dtype=INT)
        y_mask  = tensor.matrix('y_mask', dtype=FLOAT)

        # Reverse stuff
        xr      = x[::-1]
        xr_mask = x_mask[::-1]

        # Some shorthands for dimensions
        n_samples       = x.shape[1]
        n_timesteps     = x.shape[0]
        n_timesteps_trg = y.shape[0]

        # Store tensors
        self.inputs             = OrderedDict()
        self.inputs['x']        = x         # Source words
        self.inputs['x_mask']   = x_mask    # Source mask
        self.inputs['x_img']    = x_img     # Image features
        self.inputs['y']        = y         # Target labels
        self.inputs['y_mask']   = y_mask    # Target mask

        ###################
        # Source embeddings
        ###################
        # word embedding for forward rnn (source)
        emb  = dropout(self.tparams['Wemb_enc'][x.flatten()], self.trng, self.emb_dropout, self.use_dropout)
        emb  = emb.reshape([n_timesteps, n_samples, self.embedding_dim])
        forw = get_new_layer('gru')[1](self.tparams, emb, prefix='text_encoder', mask=x_mask, layernorm=self.lnorm)

        # word embedding for backward rnn (source)
        embr = dropout(self.tparams['Wemb_enc'][xr.flatten()], self.trng, self.emb_dropout, self.use_dropout)
        embr = embr.reshape([n_timesteps, n_samples, self.embedding_dim])
        back = get_new_layer('gru')[1](self.tparams, embr, prefix='text_encoder_r', mask=xr_mask, layernorm=self.lnorm)

        # Source context will be the concatenation of forward and backward rnns
        # leading to a vector of 2*rnn_dim for each timestep
        text_ctx = tensor.concatenate([forw[0], back[0][::-1]], axis=forw[0].ndim-1)
        # -> n_timesteps x n_samples x 2*rnn_dim

        # Apply dropout
        text_ctx = dropout(text_ctx, self.trng, self.ctx_dropout, self.use_dropout)

        if self.init_cgru == 'text':
            # mean of the context (across time) will be used to initialize decoder rnn
            text_ctx_mean = (text_ctx * x_mask[:, :, None]).sum(0) / x_mask.sum(0)[:, None]
            # -> n_samples x ctx_dim (2*rnn_dim)

            # initial decoder state computed from source context mean
            init_state = get_new_layer('ff')[1](self.tparams, text_ctx_mean, prefix='ff_text_state_init', activ='tanh')
            # -> n_samples x rnn_dim (last dim shrinked down by this FF to rnn_dim)
        elif self.init_cgru == 'img':
            # Reduce to nb_samples x conv_dim and transform
            init_state = get_new_layer('ff')[1](self.tparams, x_img.mean(axis=0), prefix='ff_img_state_init', activ='tanh')
        elif self.init_cgru == 'textimg':
            # n_samples x conv_dim
            img_ctx_mean  = x_img.mean(axis=0)
            # n_samples x ctx_dim
            text_ctx_mean = (text_ctx * x_mask[:, :, None]).sum(0) / x_mask.sum(0)[:, None]
            # n_samples x (conv_dim + ctx_dim)
            mmodal_ctx = tensor.concatenate([img_ctx_mean, text_ctx_mean], axis=-1)
            init_state = get_new_layer('ff')[1](self.tparams, mmodal_ctx, prefix='ff_textimg_state_init', activ='tanh')
        else:
            init_state = tensor.alloc(0., n_samples, self.rnn_dim)

        #######################
        # Source image features
        #######################

        # Project image features to ctx_dim
        img_ctx = get_new_layer('ff')[1](self.tparams, x_img, prefix='ff_img_adaptor', activ='linear')
        # -> 196 x n_samples x ctx_dim

        ####################
        # Target embeddings
        ####################

        # Fetch target embeddings. Result is: (n_trg_timesteps x n_samples x embedding_dim)
        emb_trg = self.tparams['Wemb_dec'][y.flatten()]
        emb_trg = emb_trg.reshape([n_timesteps_trg, n_samples, self.embedding_dim])

        # Shift it to right to leave place for the <bos> placeholder
        # We ignore the last word <eos> as we don't condition on it at the end
        # to produce another word
        emb_trg_shifted = tensor.zeros_like(emb_trg)
        emb_trg_shifted = tensor.set_subtensor(emb_trg_shifted[1:], emb_trg[:-1])
        emb_trg = emb_trg_shifted

        ##########
        # GRU Cond
        ##########
        # decoder - pass through the decoder conditional gru with attention
        dec_mult = self.gru_decoder(self.tparams, emb_trg,
                                    prefix='decoder_multi',
                                    input_mask=y_mask,
                                    ctx1=text_ctx, ctx1_mask=x_mask,
                                    ctx2=img_ctx,
                                    one_step=False,
                                    init_state=init_state)

        # gru_cond returns hidden state, weighted sum of context vectors and attentional weights.
        h           = dec_mult[0]    # (n_timesteps_trg, batch_size, rnn_dim)
        sumctx      = dec_mult[1]    # (n_timesteps_trg, batch_size, ctx*.shape[-1] (2000, 2*rnn_dim))
        # weights (alignment matrix)
        self.alphas = list(dec_mult[2:])

        # 3-way merge
        logit_gru  = get_new_layer('ff')[1](self.tparams, h, prefix='ff_logit_gru', activ='linear')
        logit_ctx  = get_new_layer('ff')[1](self.tparams, sumctx, prefix='ff_logit_ctx', activ='linear')

        # Dropout
        logit = dropout(tanh(logit_gru + emb_trg + logit_ctx), self.trng, self.out_dropout, self.use_dropout)

        if self.tied_trg_emb is False:
            logit = get_new_layer('ff')[1](self.tparams, logit, prefix='ff_logit', activ='linear')
        else:
            logit = tensor.dot(logit, self.tparams['Wemb_dec'].T)

        logit_shp = logit.shape

        # Apply logsoftmax (stable version)
        log_probs = -tensor.nnet.logsoftmax(logit.reshape([logit_shp[0]*logit_shp[1], logit_shp[2]]))

        # cost
        y_flat = y.flatten()
        y_flat_idx = tensor.arange(y_flat.shape[0]) * self.n_words_trg + y_flat

        cost = log_probs.flatten()[y_flat_idx]
        cost = cost.reshape([n_timesteps_trg, n_samples])
        cost = (cost * y_mask).sum(0)

        self.f_log_probs = theano.function(list(self.inputs.values()), cost)

        return cost

    def build_sampler(self):
        x               = tensor.matrix('x', dtype=INT)
        n_timesteps     = x.shape[0]
        n_samples       = x.shape[1]

        ################
        # Image features
        ################
        # 196 x 1 x 1024
        x_img           = tensor.tensor3('x_img', dtype=FLOAT)
        # Convert to 196 x 2000 (2*rnn_dim)
        img_ctx         = get_new_layer('ff')[1](self.tparams, x_img[:, 0, :], prefix='ff_img_adaptor', activ='linear')
        # Broadcast middle dimension to make it 196 x 1 x 2000
        img_ctx         = img_ctx[:, None, :]

        #####################
        # Text Bi-GRU Encoder
        #####################
        emb  = self.tparams['Wemb_enc'][x.flatten()]
        emb  = emb.reshape([n_timesteps, n_samples, self.embedding_dim])
        forw = get_new_layer('gru')[1](self.tparams, emb, prefix='text_encoder', layernorm=self.lnorm)

        embr = self.tparams['Wemb_enc'][x[::-1].flatten()]
        embr = embr.reshape([n_timesteps, n_samples, self.embedding_dim])
        back = get_new_layer('gru')[1](self.tparams, embr, prefix='text_encoder_r', layernorm=self.lnorm)

        # concatenate forward and backward rnn hidden states
        text_ctx = tensor.concatenate([forw[0], back[0][::-1]], axis=forw[0].ndim-1)

        if self.init_cgru == 'text':
            init_state = get_new_layer('ff')[1](self.tparams, text_ctx.mean(0), prefix='ff_text_state_init', activ='tanh')
        elif self.init_cgru == 'img':
            # Reduce to nb_samples x conv_dim and transform
            init_state = get_new_layer('ff')[1](self.tparams, x_img.mean(0), prefix='ff_img_state_init', activ='tanh')
        elif self.init_cgru == 'textimg':
            # n_samples x conv_dim
            img_ctx_mean  = x_img.mean(0)
            # n_samples x ctx_dim
            text_ctx_mean = text_ctx.mean(0)
            # n_samples x (conv_dim + ctx_dim)
            mmodal_ctx = tensor.concatenate([img_ctx_mean, text_ctx_mean], axis=-1)
            init_state = get_new_layer('ff')[1](self.tparams, mmodal_ctx, prefix='ff_textimg_state_init', activ='tanh')
        else:
            init_state = tensor.alloc(0., n_samples, self.rnn_dim)

        ################
        # Build f_init()
        ################
        inps        = [x, x_img]
        outs        = [init_state, text_ctx, img_ctx]
        self.f_init = theano.function(inps, outs, name='f_init')

        ###################
        # Target Embeddings
        ###################
        y       = tensor.vector('y_sampler', dtype=INT)
        emb_trg = tensor.switch(y[:, None] < 0,
                                tensor.alloc(0., 1, self.tparams['Wemb_dec'].shape[1]),
                                self.tparams['Wemb_dec'][y])

        ##########
        # Text GRU
        ##########
        dec_mult = self.gru_decoder(self.tparams, emb_trg,
                                    prefix='decoder_multi',
                                    input_mask=None,
                                    ctx1=text_ctx, ctx1_mask=None,
                                    ctx2=img_ctx,
                                    one_step=True,
                                    init_state=init_state)
        h      = dec_mult[0]
        sumctx = dec_mult[1]
        alphas = tensor.concatenate(dec_mult[2:], axis=-1)

        # 3-way merge
        logit_gru  = get_new_layer('ff')[1](self.tparams, h, prefix='ff_logit_gru', activ='linear')
        logit_ctx  = get_new_layer('ff')[1](self.tparams, sumctx, prefix='ff_logit_ctx', activ='linear')

        logit = tanh(logit_gru + emb_trg + logit_ctx)

        if self.tied_trg_emb is False:
            logit = get_new_layer('ff')[1](self.tparams, logit, prefix='ff_logit', activ='linear')
        else:
            logit = tensor.dot(logit, self.tparams['Wemb_dec'].T)

        # compute the logsoftmax
        next_log_probs = tensor.nnet.logsoftmax(logit)

        ################
        # Build f_next()
        ################
        inputs      = [y, init_state, text_ctx, img_ctx]
        outs        = [next_log_probs, h, alphas]
        self.f_next = theano.function(inputs, outs, name='f_next')
