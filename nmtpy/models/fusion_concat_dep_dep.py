from six.moves import range
from six.moves import zip

# Python
import os
import copy
import cPickle
import importlib

from collections import OrderedDict

# 3rd party
import numpy as np

import theano
import theano.tensor as tensor

# Ours
from ..typedef import *
from ..nmtutils import *
from ..iterators.wmt import WMTIterator

# Import fusion-specific CGRU
from ..layers import *
from .fusionlayers import init_multigru_concat_dep_dep
from .fusionlayers import multigru_concat_dep_dep
from .attention import Model as ParentModel

# This is the fusion model with concatenation
# and DEP-DEP attention.
init_gru_decoder_multiconcat = init_multigru_concat_dep_dep
gru_decoder_multiconcat      = multigru_concat_dep_dep


class Model(ParentModel):
    def __init__(self, seed, logger, **kwargs):
        # Call parent's init first
        super(Model, self).__init__(seed, logger, **kwargs)

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

        # convfeats (512) to ctx dim (2000) for image modality
        params = get_new_layer('ff')[0](params, prefix='ff_img_adaptor', nin=self.conv_dim, nout=self.ctx_dim, scale=self.weight_init)

        #############################################
        # Source sentence encoder: bidirectional GRU
        #############################################
        # Forward and backward encoder parameters
        params = get_new_layer('gru')[0](params, prefix='text_encoder'  , nin=self.embedding_dim, dim=self.rnn_dim, scale=self.weight_init)
        params = get_new_layer('gru')[0](params, prefix='text_encoder_r', nin=self.embedding_dim, dim=self.rnn_dim, scale=self.weight_init)

        ##########
        # Decoder
        ##########
        # init_state computation from mean context: 2000 -> 1000 if rnn_dim == 1000
        params = get_new_layer('ff')[0](params, prefix='ff_text_state_init', nin=self.ctx_dim, nout=self.rnn_dim, scale=self.weight_init)

        # GRU cond decoder
        params = init_gru_decoder_multiconcat(params, prefix='decoder_multi', nin=self.embedding_dim,
                                              dim=self.rnn_dim, dimctx=self.ctx_dim, scale=self.weight_init)

        # readout
        # NOTE: In the text NMT, we also have logit_prev that is applied onto emb_trg
        # NOTE: ortho= changes from text NMT to SAT. Need to experiment
        params = get_new_layer('ff')[0](params, prefix='ff_logit_gru', nin=self.rnn_dim     , nout=self.embedding_dim, scale=self.weight_init)
        params = get_new_layer('ff')[0](params, prefix='ff_logit_ctx', nin=self.ctx_dim     , nout=self.embedding_dim, scale=self.weight_init)
        params = get_new_layer('ff')[0](params, prefix='ff_logit'    , nin=self.embedding_dim , nout=self.n_words_trg, scale=self.weight_init)

        # Save initial parameters for debugging purposes
        self.initial_params = params

    def build(self):
        # Source sentences: n_timesteps, n_samples
        x = tensor.matrix('x', dtype=INT)
        x_mask = tensor.matrix('x_mask', dtype=FLOAT)

        # Image: 196 (n_annotations) x n_samples x 512 (ctxdim)
        x_img = tensor.tensor3('x_img', dtype=FLOAT)

        # Target sentences: n_timesteps, n_samples
        y = tensor.matrix('y', dtype=INT)
        y_mask = tensor.matrix('y_mask', dtype=FLOAT)

        # Some shorthands for dimensions
        n_samples       = x.shape[1]
        n_timesteps     = x.shape[0]
        n_timesteps_trg = y.shape[0]

        # Store tensors
        self.inputs['x']        = x         # Source words
        self.inputs['x_mask']   = x_mask    # Source mask
        self.inputs['x_img']    = x_img     # Image features
        self.inputs['y']        = y         # Target labels
        self.inputs['y_mask']   = y_mask    # Target mask

        ###################
        # Source embeddings
        ###################
        # Fetch source embeddings. Result is: (n_timesteps x n_samples x embedding_dim)
        emb_enc = self.tparams['Wemb_enc'][x.flatten()].reshape([n_timesteps, n_samples, self.embedding_dim])
        # -> n_timesteps x n_samples x embedding_dim

        # Pass the source word vectors through the GRU RNN
        emb_enc_rnns = get_new_layer('gru')[1](self.tparams, emb_enc, prefix='text_encoder', mask=x_mask)
        # -> n_timesteps x n_samples x rnn_dim

        # word embedding for backward rnn (source)
        # for the backward rnn, we just need to invert x and x_mask
        xr      = x[::-1]
        xr_mask = x_mask[::-1]
        emb_enc_r = self.tparams['Wemb_enc'][xr.flatten()].reshape([n_timesteps, n_samples, self.embedding_dim])
        # -> n_timesteps x n_samples x embedding_dim
        # Pass the source word vectors in reverse through the GRU RNN
        emb_enc_rnns_r = get_new_layer('gru')[1](self.tparams, emb_enc_r, prefix='text_encoder_r', mask=xr_mask)
        # -> n_timesteps x n_samples x rnn_dim

        # Source context will be the concatenation of forward and backward rnns
        # leading to a vector of 2*rnn_dim for each timestep
        text_ctx = tensor.concatenate([emb_enc_rnns[0], emb_enc_rnns_r[0][::-1]], axis=emb_enc_rnns[0].ndim-1)
        # -> n_timesteps x n_samples x 2*rnn_dim

        # mean of the context (across time) will be used to initialize decoder rnn
        text_ctx_mean = (text_ctx * x_mask[:, :, None]).sum(0) / x_mask.sum(0)[:, None]
        # -> n_samples x ctx_dim (2*rnn_dim)

        # initial decoder state computed from source context mean
        # NOTE: Can the two initializer be merged into one?
        text_init_state = get_new_layer('ff')[1](self.tparams, text_ctx_mean, prefix='ff_text_state_init', activ='tanh')
        # -> n_samples x rnn_dim (last dim shrinked down by this FF to rnn_dim)

        #######################
        # Source image features
        #######################

        # Project image features to ctx_dim
        img_ctx = get_new_layer('ff')[1](self.tparams, x_img, prefix='ff_img_adaptor', activ='linear')
        # -> 196 x n_samples x ctx_dim

        # initial decoder state learned from mean image context
        # NOTE: Can the two initializer be merged into one?
        #img_init_state = get_new_layer('ff')[1](self.tparams, img_ctx.mean(0), prefix='ff_img_state_init', activ='tanh')
        # -> n_samples x rnn_dim

        ####################
        # Target embeddings
        ####################

        # Fetch target embeddings. Result is: (n_trg_timesteps x n_samples x embedding_dim)
        emb_trg = self.tparams['Wemb_dec'][y.flatten()].reshape([y.shape[0], y.shape[1], self.embedding_dim])

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
        dec_mult = gru_decoder_multiconcat(self.tparams, emb_trg,
                                           prefix='decoder_multi',
                                           input_mask=y_mask,
                                           ctx1=text_ctx, ctx1_mask=x_mask,
                                           ctx2=img_ctx,
                                           one_step=False,
                                           init_state=text_init_state) # NOTE: init_state only text

        # gru_cond returns hidden state, weighted sum of context vectors and attentional weights.
        h       = dec_mult[0]    # (n_timesteps_trg, batch_size, rnn_dim)
        sumctx  = dec_mult[1]    # (n_timesteps_trg, batch_size, ctx*.shape[-1] (2000, 2*rnn_dim))

        self.alphas  = list(dec_mult[2:])

        logit    = emb_trg
        logit   += get_new_layer('ff')[1](self.tparams, h, prefix='ff_logit_gru', activ='linear')
        logit   += get_new_layer('ff')[1](self.tparams, sumctx, prefix='ff_logit_ctx', activ='linear')

        # tanh over logit
        logit = tanh(logit)


        # embedding_dim -> n_words_trg
        logit = get_new_layer('ff')[1](self.tparams, logit, prefix='ff_logit', activ='linear')
        logit_shp = logit.shape

        # Apply logsoftmax (stable version)
        log_probs = -tensor.nnet.logsoftmax(logit.reshape([logit_shp[0]*logit_shp[1], logit_shp[2]]))

        # cost
        y_flat = y.flatten()
        y_flat_idx = tensor.arange(y_flat.shape[0]) * self.n_words_trg + y_flat

        cost = log_probs.flatten()[y_flat_idx]
        cost = cost.reshape([y.shape[0], y.shape[1]])
        cost = (cost * y_mask).sum(0)

        self.f_log_probs = theano.function(self.inputs.values(), cost)

        if self.norm_cost:
            return (cost / y_mask.sum(0)).mean()
        else:
            return cost.mean()

    def build_sampler(self):
        x               = tensor.matrix('x', dtype=INT)
        n_timesteps     = x.shape[0]
        n_samples       = x.shape[1]

        ################
        # Image features
        ################
        # 196 x 512
        x_img           = tensor.matrix('x_img', dtype=FLOAT)
        # Convert to 196 x 2000 (2*rnn_dim)
        img_ctx         = get_new_layer('ff')[1](self.tparams, x_img, prefix='ff_img_adaptor', activ='linear')
        # Broadcast middle dimension to make it 196 x 1 x 2000
        img_ctx         = img_ctx[:, None, :]
        # Take the mean over the first dimension: 1 x 2000
        img_ctx_mean    = img_ctx.mean(0)
        # Give the mean to compute the initial state: 1 x 1000
        #img_init_state  = get_new_layer('ff')[1](self.tparams, img_ctx_mean, prefix='ff_img_state_init', activ='tanh')

        #####################
        # Text Bi-GRU Encoder
        #####################
        emb             = self.tparams['Wemb_enc'][x.flatten()]
        emb             = emb.reshape([n_timesteps, n_samples, self.embedding_dim])
        proj            = get_new_layer('gru')[1](self.tparams, emb, prefix='text_encoder')

        embr            = self.tparams['Wemb_enc'][x[::-1].flatten()]
        embr            = embr.reshape([n_timesteps, n_samples, self.embedding_dim])
        projr           = get_new_layer('gru')[1](self.tparams, embr, prefix='text_encoder_r')

        # concatenate forward and backward rnn hidden states
        text_ctx        = tensor.concatenate([proj[0], projr[0][::-1]], axis=proj[0].ndim-1)

        # get the input for decoder rnn initializer mlp
        text_ctx_mean   = text_ctx.mean(0)
        text_init_state = get_new_layer('ff')[1](self.tparams, text_ctx_mean, prefix='ff_text_state_init', activ='tanh')

        ################
        # Build f_init()
        ################
        inps            = [x, x_img]
        outs            = [text_ctx, img_ctx, text_init_state]
        self.f_init     = theano.function(inps, outs, name='f_init')

        ###################
        # Target Embeddings
        ###################
        y               = tensor.vector('y_sampler', dtype=INT)
        emb             = tensor.switch(y[:, None] < 0,
                                        tensor.alloc(0., 1, self.tparams['Wemb_dec'].shape[1]),
                                        self.tparams['Wemb_dec'][y])

        ##########
        # Text GRU
        ##########
        dec_mult = gru_decoder_multiconcat(self.tparams, emb,
                                           prefix='decoder_multi',
                                           input_mask=None,
                                           ctx1=text_ctx, ctx1_mask=None,
                                           ctx2=img_ctx,
                                           one_step=True,
                                           init_state=text_init_state)
        h      = dec_mult[0]
        sumctx = dec_mult[1]
        alphas = list(dec_mult[2:])

        ########
        # Fusion
        ########
        logit       = emb
        logit       += get_new_layer('ff')[1](self.tparams, sumctx, prefix='ff_logit_ctx', activ='linear')
        logit       += get_new_layer('ff')[1](self.tparams, h, prefix='ff_logit_gru', activ='linear')
        logit = tanh(logit)


        logit = get_new_layer('ff')[1](self.tparams, logit, prefix='ff_logit', activ='linear')

        # compute the logsoftmax
        next_log_probs = tensor.nnet.logsoftmax(logit)

        # Sample from the softmax distribution
        next_probs = tensor.exp(next_log_probs)
        next_word = self.trng.multinomial(pvals=next_probs).argmax(1)

        ################
        # Build f_next()
        ################
        inputs = [y, text_ctx, img_ctx, text_init_state]
        outs = [next_log_probs, next_word, h] + alphas
        self.f_next = theano.function(inputs, outs, name='f_next')

    def beam_search(self, inputs, beam_size=12, maxlen=50, suppress_unks=False, **kwargs):
        get_att = kwargs.get('get_att_alphas', False)

        # Final results and their scores
        final_sample        = []
        final_score         = []
        final_alignments    = []

        # Initially we have one empty hypothesis with a score of 0
        hyp_alignments  = [[]]
        hyp_samples     = [[]]
        hyp_scores      = np.zeros(1, dtype=FLOAT)

        # get initial state of decoder rnn and encoder context vectors
        # ctx0: the set of context vectors leading to the next_state
        # with an initial shape of (n_src_words x 1 x ctx_dim)

        text_ctx, img_ctx, next_state = self.f_init(*inputs)

        # Beginning-of-sentence indicator is -1
        next_w = -1 * np.ones((1,), dtype=INT)

        # maxlen or 3 times source length
        maxlen = min(maxlen, inputs[0].shape[0] * 3)

        # Always starts with the initial tstep's context vectors
        # e.g. we have a ctx0 of shape (n_words x 1 x ctx_dim)
        # Tiling it live_beam times makes it (n_words x live_beam x ctx_dim)
        # thus we create sth like a batch of live_beam size with every word duplicated
        # for further state expansion.
        tiled_ctx = np.tile(text_ctx, [1, 1])
        live_beam = beam_size

        for ii in xrange(maxlen):
            # Get next states
            # In the first iteration, we provide -1 and obtain the log_p's for the
            # first word. In the following iterations tiled_ctx becomes a batch
            # of duplicated left hypotheses. tiled_ctx is always the same except
            # the size of the 2nd dimension as the context vectors of the source
            # sequence is always the same regardless of the decoding process.
            # next_state's shape is (live_beam, rnn_dim)
            next_log_p, _, next_state, alpha_txt, alpha_img = self.f_next(next_w, tiled_ctx, img_ctx, next_state)

            # For each f_next, we obtain a new set of alpha's for the next_w
            # for each hypothesis in the beam search

            if suppress_unks:
                next_log_p[:, 1] = -np.inf

            # Compute sum of log_p's for the current hypotheses
            cand_scores = hyp_scores[:, None] - next_log_p

            # Flatten by modifying .shape (faster)
            cand_scores.shape = cand_scores.size

            # Take the best live_beam hypotheses
            # argpartition makes a partial sort which is faster than argsort
            # (Idea taken from https://github.com/rsennrich/nematus)
            ranks_flat = cand_scores.argpartition(live_beam-1)[:live_beam]

            # Get the costs
            costs = cand_scores[ranks_flat]

            # New states, scores and samples
            live_beam           = 0
            new_hyp_scores      = []
            new_hyp_samples     = []
            new_hyp_alignments  = []

            # This will be the new next states in the next iteration
            hyp_states          = []

            # Find out to which initial hypothesis idx this was belonging
            # Find out the idx of the appended word
            trans_idxs  = ranks_flat / self.n_words_trg
            word_idxs   = ranks_flat % self.n_words_trg
            # Iterate over the hypotheses and add them to new_* lists
            for idx, [ti, wi] in enumerate(zip(trans_idxs, word_idxs)):
                # Form the new hypothesis by appending new word to the left hyp
                new_hyp = hyp_samples[ti] + [wi]
                new_ali = hyp_alignments[ti] + [[alpha_txt[ti], alpha_img[ti]]]

                if wi == 0:
                    # <eos> found, separate out finished hypotheses
                    final_sample.append(new_hyp)
                    final_score.append(costs[idx])
                    final_alignments.append(new_ali)
                else:
                    # Add formed hypothesis to the new hypotheses list
                    new_hyp_samples.append(new_hyp)
                    # Cumulated cost of this hypothesis
                    new_hyp_scores.append(costs[idx])
                    # Hidden state of the decoder for this hypothesis
                    hyp_states.append(next_state[ti])
                    new_hyp_alignments.append(new_ali)
                    live_beam += 1

            hyp_scores  = np.array(new_hyp_scores, dtype=FLOAT)
            hyp_samples = new_hyp_samples
            hyp_alignments = new_hyp_alignments

            if live_beam == 0:
                break

            # Take the idxs of each hyp's last word
            next_w      = np.array([w[-1] for w in hyp_samples])
            next_state  = np.array(hyp_states, dtype=FLOAT)
            tiled_ctx   = np.tile(text_ctx, [live_beam, 1])

        # dump every remaining hypotheses
        for idx in xrange(live_beam):
            final_sample.append(hyp_samples[idx])
            final_score.append(hyp_scores[idx])
            final_alignments.append(hyp_alignments[idx])

        if get_att:
            return final_sample, final_score, final_alignments
        else:
            return final_sample, final_score

    def get_alpha_regularizer(self, alpha_c):
        alpha_c = theano.shared(np.float32(alpha_c), name='alpha_c')
        alpha_reg = alpha_c * ((1.-self.alphas[1].sum(0))**2).sum(0).mean()
        return alpha_reg