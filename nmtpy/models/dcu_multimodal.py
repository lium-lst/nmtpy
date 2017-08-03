# -*- coding: utf-8 -*-
from collections import OrderedDict
import numpy as np

import theano
import theano.tensor as tensor

# Ours
from ..layers import *
from ..nmtutils import norm_weight
from ..defaults import INT, FLOAT
from ..iterators.wmt import WMTIterator

# Base fusion model
from .attention import Model as Attention

def init_gru_decoder_multi(params, nin, dim, dimctx, scale=0.01, prefix='gru_decoder_multi'):
    # Init with usual gru_cond function
    params = param_init_gru_cond(params, nin, dim, dimctx, scale, prefix)

    # attention: This gives the alpha's
    params[pp(prefix, 'U_att2')]        = norm_weight(dimctx, 1, scale=scale)
    params[pp(prefix, 'c_att2')]        = np.zeros((1,)).astype(FLOAT)
    params[pp(prefix, 'W_comb_att2')]   = norm_weight(dim, dimctx, scale=scale)

    params[pp(prefix, 'W2_txt')] = np.concatenate([norm_weight(dimctx, dim, scale=scale),
                                                   norm_weight(dimctx, dim, scale=scale)], axis=1)
    params[pp(prefix, 'W2_img')] = np.concatenate([norm_weight(dimctx, dim, scale=scale),
                                                   norm_weight(dimctx, dim, scale=scale)], axis=1)
    params[pp(prefix, 'W3_txt')] = norm_weight(dimctx, dim, scale=scale)
    params[pp(prefix, 'W3_img')] = norm_weight(dimctx, dim, scale=scale)
    params[pp(prefix, 'W_sel')]  = norm_weight(dim, 1, scale=scale)
    # NOTE: Gate may start as half open
    params[pp(prefix, 'b_sel')]  = np.zeros((1,)).astype(FLOAT)

    # Clean previously allocated stuff
    for param in ['b_nl', 'bx_nl', 'Wc', 'Wcx']:
        del params[pp(prefix, param)]
    return params

def gru_decoder_multi(tparams, state_below,
                      ctx1, ctx2, prefix='gru_decoder_multi',
                      input_mask=None, one_step=False,
                      init_state=None, ctx1_mask=None):
    if one_step:
        assert init_state, 'previous state must be provided'

    # Context
    # n_timesteps x n_samples x ctxdim
    assert ctx1 and ctx2, 'Contexts must be provided'
    assert ctx1.ndim == 3 and ctx2.ndim == 3, 'Contexts must be 3-d: #annotation x #sample x dim'

    # Number of padded source timesteps
    nsteps = state_below.shape[0]

    # Batch or single sample?
    n_samples = state_below.shape[1] if state_below.ndim == 3 else 1

    # if we have no mask, we assume all the inputs are valid
    # tensor.alloc(value, *shape)
    # input_mask: (n_steps, 1) filled with 1
    if input_mask is None:
        input_mask = tensor.alloc(1., nsteps, 1)

    # Infer RNN dimensionality
    dim = tparams[pp(prefix, 'W_comb_att2')].shape[0]

    # initial/previous state
    # if not given, assume it's all zeros
    if init_state is None:
        init_state = tensor.alloc(0., n_samples, dim)

    # These two dot products are same with gru_layer, refer to the equations.
    # [W_r * X + b_r, W_z * X + b_z]
    state_below_ = tensor.dot(state_below, tparams[pp(prefix, 'W')]) + tparams[pp(prefix, 'b')]

    # input to compute the hidden state proposal
    # This is the [W*x]_j in the eq. 8 of the paper
    state_belowx = tensor.dot(state_below, tparams[pp(prefix, 'Wx')]) + tparams[pp(prefix, 'bx')]

    # Wc_att: dimctx -> dimctx
    # Linearly transform the contexts to another space with same dimensionality
    pctx1_ = tensor.dot(ctx1, tparams[pp(prefix, 'Wc_att')]) + tparams[pp(prefix, 'b_att')]
    # Do not transform image context again, it was already transformed by img_adaptor

    # Step function for the recurrence/scan
    # Sequences
    # ---------
    # m_    : mask
    # x_    : state_below_
    # xx_   : state_belowx
    # outputs_info
    # ------------
    # h_     : init_state,
    # ctx_   : need to be defined as it's returned by _step
    # alpha1_: need to be defined as it's returned by _step
    # alpha2_: need to be defined as it's returned by _step
    # non sequences
    # -------------
    # pctx1_ : pctx1_
    # cc1_   : ctx1
    # cc2_   : ctx2
    # and all the shared weights and biases..
    def _step(m_, x_, xx_,
              h_, c_t, i_t, alpha1_, alpha2_, # These ctx and alpha's are not used in the computations
              pctx1_, cc1_, cc2_, U, W_comb_att, W_comb_att2,
              U_att, c_att, U_att2, c_att2,
              Ux, U_nl, Ux_nl, W2_txt, W2_img, W3_txt, W3_img, W_sel, b_sel):

        # Do a step of classical GRU
        h1 = gru_step(m_, x_, xx_, h_, U, Ux)

        ######################################
        # Distinct attention for each modality
        ######################################
        # h1 X W_comb_att
        # W_comb_att: dim -> dimctx
        # pstate_ should be 2D as we're working with unrolled timesteps
        pstate1_ = tensor.dot(h1, W_comb_att)
        pstate2_ = tensor.dot(h1, W_comb_att2)

        # Accumulate in pctx*__ and apply tanh()
        # This becomes the projected context(s) + the current hidden state
        # of the decoder, e.g. this is the information accumulating
        # into the returned original contexts with the knowledge of target
        # sentence decoding.
        pctx1__ = tanh(pctx1_ + pstate1_[None, :, :])
        pctx2__ = tanh(cc2_   + pstate2_[None, :, :])

        # Affine transformation for alpha* = (pctx*__ X U_att) + c_att
        # We're now down to scalar alpha's for each accumulated
        # context (0th dim) in the pctx*__
        # alpha1 should be n_timesteps, 1, 1
        alpha1 = tensor.dot(pctx1__, U_att) + c_att
        alpha2 = tensor.dot(pctx2__, U_att2) + c_att2

        # Drop the last dimension, e.g. (n_timesteps, 1)
        alpha1 = alpha1.reshape([alpha1.shape[0], alpha1.shape[1]])
        alpha2 = alpha2.reshape([alpha2.shape[0], alpha2.shape[1]])

        # Exponentiate alpha1
        alpha1 = tensor.exp(alpha1 - alpha1.max(0, keepdims=True))
        alpha2 = tensor.exp(alpha2 - alpha2.max(0, keepdims=True))

        # If there is a context mask, multiply with it to cancel unnecessary steps
        # We won't have a ctx_mask for image vectors
        if ctx1_mask:
            alpha1 = alpha1 * ctx1_mask

        # Normalize so that the sum makes 1
        alpha1 = alpha1 / alpha1.sum(0, keepdims=True)
        alpha2 = alpha2 / alpha2.sum(0, keepdims=True)

        # Compute the current context ctx*_ as the alpha-weighted sum of
        # the initial contexts ctx*'s
        c_t = (cc1_ * alpha1[:, :, None]).sum(0)
        i_t = (cc2_ * alpha2[:, :, None]).sum(0)
        # n_samples x ctxdim (2000)

        # Apply selector based on the previous hidden state of CGRU
        sel_t = sigmoid(tensor.dot(h_, W_sel) + b_sel)
        # Let it broadcast
        i_t = i_t * sel_t[:, 0][:, None]

        ####################################
        # The below code is another GRU cell
        ####################################
        # h1: s't
        z_and_r = sigmoid(tensor.dot(h1, U_nl) + tensor.dot(c_t, W2_txt) + tensor.dot(i_t, W2_img))

        # Gather specific gates (old r -> z2, old u -> r2)
        z2 = tensor_slice(z_and_r, 0, dim)
        r2 = tensor_slice(z_and_r, 1, dim)

        # Candidate hidden h2_tilda: s_bar_t
        h2_tilda = tanh(tensor.dot(c_t, W3_txt) + tensor.dot(i_t, W3_img) + (tensor.dot(h1, Ux_nl) * r2))

        # Leaky integration between the new h2 and the old h1 computed
        h2 = z2 * h2_tilda + (1. - z2) * h1
        h2 = m_[:, None] * h2 + (1. - m_)[:, None] * h1

        # No fusion of contexts
        return h2, c_t, i_t, alpha1.T, alpha2.T

    # Sequences are the input mask and the transformed target embeddings
    seqs = [input_mask, state_below_, state_belowx]

    # Create a list of shared parameters for easy parameter passing
    shared_vars = [
                    tparams[pp(prefix, 'U')],
                    tparams[pp(prefix, 'W_comb_att')],
                    tparams[pp(prefix, 'W_comb_att2')],
                    tparams[pp(prefix, 'U_att')],
                    tparams[pp(prefix, 'c_att')],
                    tparams[pp(prefix, 'U_att2')],
                    tparams[pp(prefix, 'c_att2')],
                    tparams[pp(prefix, 'Ux')],
                    tparams[pp(prefix, 'U_nl')],
                    tparams[pp(prefix, 'Ux_nl')],
                    tparams[pp(prefix, 'W2_txt')],
                    tparams[pp(prefix, 'W2_img')],
                    tparams[pp(prefix, 'W3_txt')],
                    tparams[pp(prefix, 'W3_img')],
                    tparams[pp(prefix, 'W_sel')],
                    tparams[pp(prefix, 'b_sel')],
                  ]

    if one_step:
        rval = _step(*(seqs + [init_state, None, None, None, None, pctx1_, ctx1, ctx2] + shared_vars))
    else:
        outputs_info=[init_state,
                      tensor.alloc(0., n_samples, ctx1.shape[2]), # ctxdim       (c_t)
                      tensor.alloc(0., n_samples, ctx2.shape[2]), # ctxdim       (i_t)
                      tensor.alloc(0., n_samples, ctx1.shape[0]), # n_timesteps  (alpha1)
                      tensor.alloc(0., n_samples, ctx2.shape[0])] # n_timesteps  (alpha2)

        rval, updates = theano.scan(_step,
                                    sequences=seqs,
                                    outputs_info=outputs_info,
                                    non_sequences=[pctx1_, ctx1, ctx2] + shared_vars,
                                    name=pp(prefix, '_layers'),
                                    n_steps=nsteps,
                                    strict=True)
    return rval

class Model(Attention):
    def __init__(self, seed, logger, **kwargs):
        # Call Attention's __init__
        super(Model, self).__init__(seed, logger, **kwargs)

        self.init_gru_decoder   = init_gru_decoder_multi
        self.gru_decoder        = gru_decoder_multi

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
        params = get_new_layer('ff')[0](params, prefix='ff_logit_ctx_text', nin=self.ctx_dim,
                                        nout=self.embedding_dim, scale=self.weight_init)
        params = get_new_layer('ff')[0](params, prefix='ff_logit_ctx_img', nin=self.ctx_dim,
                                        nout=self.embedding_dim, scale=self.weight_init)
        params = get_new_layer('ff')[0](params, prefix='ff_logit_emb', nin=self.embedding_dim,
                                        nout=self.embedding_dim, scale=self.weight_init)
        if self.tied_emb is False:
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
        c_t         = dec_mult[1]    # (n_timesteps_trg, batch_size, ctx*.shape[-1] (2000, 2*rnn_dim))
        i_t         = dec_mult[2]    # (n_timesteps_trg, batch_size, ctx*.shape[-1] (2000, 2*rnn_dim))
        # weights (alignment matrix)
        self.alphas = list(dec_mult[3:])

        # 3-way merge
        logit_gru       = get_new_layer('ff')[1](self.tparams, h, prefix='ff_logit_gru', activ='linear')
        logit_ctx_text  = get_new_layer('ff')[1](self.tparams, c_t, prefix='ff_logit_ctx_text', activ='linear')
        logit_ctx_img   = get_new_layer('ff')[1](self.tparams, i_t, prefix='ff_logit_ctx_img', activ='linear')
        logit_emb       = get_new_layer('ff')[1](self.tparams, emb_trg, prefix='ff_logit_emb', activ='linear')

        # Dropout
        logit = dropout(tanh(logit_gru + logit_emb + logit_ctx_text + logit_ctx_img), self.trng, self.out_dropout, self.use_dropout)

        if self.tied_emb is False:
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

    def build_sampler(self, **kwargs):
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
        h       = dec_mult[0]
        c_t     = dec_mult[1]
        i_t     = dec_mult[2]
        alphas = tensor.concatenate(dec_mult[3:], axis=-1)

        # 3-way merge
        logit_gru       = get_new_layer('ff')[1](self.tparams, h, prefix='ff_logit_gru', activ='linear')
        logit_ctx_text  = get_new_layer('ff')[1](self.tparams, c_t, prefix='ff_logit_ctx_text', activ='linear')
        logit_ctx_img   = get_new_layer('ff')[1](self.tparams, i_t, prefix='ff_logit_ctx_img', activ='linear')
        logit_emb       = get_new_layer('ff')[1](self.tparams, emb_trg, prefix='ff_logit_emb', activ='linear')

        # NOTE: They also had logit_emb
        logit = tanh(logit_gru + logit_emb + logit_ctx_text + logit_ctx_img)

        if self.tied_emb is False:
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
