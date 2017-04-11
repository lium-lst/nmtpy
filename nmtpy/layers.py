# -*- coding: utf-8 -*-
import numpy as np

import theano
from theano import tensor

from .nmtutils import norm_weight, ortho_weight, pp
from .defaults import FLOAT

# Shorthands for activations
linear  = lambda x: x
sigmoid = tensor.nnet.sigmoid
tanh    = tensor.tanh
relu    = tensor.nnet.relu

# Slice a tensor
def tensor_slice(_x, n, dim):
    if _x.ndim == 3:
        return _x[:, :, n*dim:(n+1)*dim]
    elif _x.ndim == 2:
        return _x[:, n*dim:(n+1)*dim]
    return _x[n*dim:(n+1)*dim]

###############################################
# Returns the initializer and the layer itself
###############################################
def get_new_layer(name):
    # Layer type: (initializer, layer)
    layers = {
                # Highway layer
                'highway'           : ('param_init_hiway'       , 'hiway_layer'),
                # Convolutional layer (not-tested)
                'conv'              : ('param_init_conv'        , 'conv_layer'),
                # Feedforward Layer
                'ff'                : ('param_init_fflayer'     , 'fflayer'),
                # GRU
                'gru'               : ('param_init_gru'         , 'gru_layer'),
                # Conditional GRU
                'gru_cond'          : ('param_init_gru_cond'    , 'gru_cond_layer'),
                # LSTM
                'lstm'              : ('param_init_lstm'        , 'lstm_layer'),
             }

    init, layer = layers[name]
    return (eval(init), eval(layer))

#################
# Forward dropout
#################
def dropout(x, rng, p, in_train):
    if p <= 0.:
        # No dropout, return x intact
        return x

    success = 1. - p
    return tensor.switch(in_train,     # If True, next one else last one
                         x * rng.binomial(x.shape, p=success, dtype=x.dtype) / success,
                         x)

#############################################################################
# Layer normalization
# Lei Ba, Jimmy, Jamie Ryan Kiros, and Geoffrey E. Hinton.
# "Layer Normalization." arXiv preprint arXiv:1607.06450 (2016).
# https://github.com/ryankiros/layer-norm
def layer_norm(x, b, s, eps=1e-5):
    output = (x - x.mean(1)[:, None]) / tensor.sqrt(x.var(1)[:, None] + eps)
    output = s[None, :] * output + b[None, :]
    return output

def init_layer_norm(prefix, params, dim, scale_add=0.0, scale_mul=1.0):
    params[pp(prefix,'b1')] = scale_add * np.ones((2*dim)).astype(FLOAT)
    params[pp(prefix,'b2')] = scale_add * np.ones((1*dim)).astype(FLOAT)
    params[pp(prefix,'b3')] = scale_add * np.ones((2*dim)).astype(FLOAT)
    params[pp(prefix,'b4')] = scale_add * np.ones((1*dim)).astype(FLOAT)
    params[pp(prefix,'s1')] = scale_mul * np.ones((2*dim)).astype(FLOAT)
    params[pp(prefix,'s2')] = scale_mul * np.ones((1*dim)).astype(FLOAT)
    params[pp(prefix,'s3')] = scale_mul * np.ones((2*dim)).astype(FLOAT)
    params[pp(prefix,'s4')] = scale_mul * np.ones((1*dim)).astype(FLOAT)

    return params

# GRU step with Layer Normalization
# Same code as below but with layer_norm addition
def gru_step_lnorm(m_, x_, xx_, h_, U, Ux, b1, b2, b3, b4, s1, s2, s3, s4):
    dim = Ux.shape[1]

    # Normalize inputs
    x_  = layer_norm(x_, b1, s1)
    xx_ = layer_norm(xx_, b2, s2)

    # Normalize dot product
    preact = sigmoid(layer_norm(tensor.dot(h_, U), b3, s3) + x_)

    r = tensor_slice(preact, 0, dim)
    u = tensor_slice(preact, 1, dim)

    # Normalize dot product
    h_tilda = tanh((layer_norm(tensor.dot(h_, Ux), b4, s4) * r) + xx_)

    h = u * h_tilda + (1. - u) * h_
    h = m_[:, None] * h + (1. - m_)[:, None] * h_

    return h
#############################################################################

##################
# GRU layer step()
##################
# sequences:
#   m_    : mask
#   x_    : state_below_
#   xx_   : state_belowx
# outputs-info:
#   h_    : init_states
# non-seqs:
#   U     : shared U matrix
#   Ux    : shared Ux matrix
def gru_step(m_, x_, xx_, h_, U, Ux):
    dim = Ux.shape[1]

    # sigmoid([U_r * h_ + (W_r * X + b_r) , U_z * h_ + (W_z * X + b_z)])
    preact = sigmoid(tensor.dot(h_, U) + x_)

    # slice reset and update gates
    r = tensor_slice(preact, 0, dim)
    u = tensor_slice(preact, 1, dim)

    # NOTE: Is this correct or should be tensor.dot(h_ * r, Ux) ?
    # hidden state proposal (h_tilda_j eq. 8)
    h_tilda = tanh(((tensor.dot(h_, Ux)) * r) + xx_)

    # leaky integrate and obtain next hidden state
    # According to paper, this should be [h = u * h_tilda + (1 - u) * h_]
    h = u * h_tilda + (1. - u) * h_
    # -> h is new h if mask is not 0 (a word was presented), otherwise, h is the copy of previous h which is h_
    h = m_[:, None] * h + (1. - m_)[:, None] * h_

    return h

#####################
# Convolutional layer
#####################
# TODO: Make the interface more easy
# TODO: Rework initialization (keras)
def param_init_conv(params, input_shape, filter_shape, scale='he', prefix='conv'):
    # input_shape : (input_channels, input_rows, input_cols)
    # filter_shape: (output_channels, input_channels, filter_rows, filter_cols)
    n_inp_chan, n_inp_row, n_in_col = input_shape
    n_out_chan, n_inp_chan, n_filt_row, n_filt_col = filter_shape

    W = norm_weight(n_filt_row*n_filt_col*n_inp_chan, n_out_chan, scale=scale)
    # Conv layer weights as 4D tensor
    params[pp(prefix, 'W')] = W.reshape((n_out_chan, n_inp_chan, n_filt_row, n_filt_col))
    # 1 bias per output channel
    params[pp(prefix, 'b')] = np.zeros((n_out_chan, )).astype(FLOAT)

    return params

def conv_layer(tparams, state_below, prefix='conv', activ='relu'):
    # state_below shape should be bc01
    return eval(activ) (
        tensor.nnet.conv2d(state_below, tparams[pp(prefix, 'W')], border_mode='valid') +
        tparams[pp(prefix, 'b')][None, :, None, None]
        )

#####################################################################
# feedforward layer: affine transformation + point-wise nonlinearity
#####################################################################
def param_init_fflayer(params, nin, nout, scale=0.01, ortho=True, prefix='ff'):
    params[pp(prefix, 'W')] = norm_weight(nin, nout, scale=scale, ortho=ortho)
    params[pp(prefix, 'b')] = np.zeros((nout,)).astype(FLOAT)

    return params

def fflayer(tparams, state_below, prefix='ff', activ='tanh'):
    return eval(activ) (
        tensor.dot(state_below, tparams[pp(prefix, 'W')]) +
        tparams[pp(prefix, 'b')]
        )

##################################
# Highway layer
# https://arxiv.org/abs/1505.00387
##################################
def param_init_hiway(params, nin, scale=0.01, ortho=True, prefix='hw', carry_bias=-1.0):
    # nin == nout for HW layers
    params[pp(prefix, 'W_t')] = norm_weight(nin, nin, scale=scale, ortho=ortho)
    params[pp(prefix, 'b_t')] = carry_bias * np.ones((nin,)).astype(FLOAT)

    params[pp(prefix, 'W')]   = norm_weight(nin, nin, scale=scale, ortho=ortho)
    params[pp(prefix, 'b')]   = np.zeros((nin,)).astype(FLOAT)

    return params

def hiwaylayer(tparams, x, prefix='hw', activ='tanh'):
    # Transform gate(x, W_t, b_t) = sigmoid(W_t * x + b_t)
    t = sigmoid(tensor.dot(x, tparams[pp(prefix, 'W_t')]) + tparams[pp(prefix, 'b_t')])

    # Normal affine transform + non-linearity
    h = eval(activ)(tensor.dot(x, tparams[pp(prefix, 'W')]) + tparams[pp(prefix, 'b')])

    # Right hand side defines input to be carried
    # while left hand side is classical FF
    out = (t * h) + (1. - t) * x

    return out

###########
# GRU layer
###########
def param_init_gru(params, nin, dim, scale=0.01, prefix='gru', layernorm=False):
    """Initialize parameters for a GRU layer. If layernorm is True, add additional
    parameters for layer normalization."""
    # See the paper for variable names
    # W is stacked W_r and W_z
    params[pp(prefix, 'W')]  = np.concatenate([norm_weight(nin, dim, scale=scale),
                                               norm_weight(nin, dim, scale=scale)], axis=1)
    # b_r and b_z
    params[pp(prefix, 'b')]  = np.zeros((2 * dim,)).astype(FLOAT)

    # recurrent transformation weights for gates
    # U is stacked U_r and U_z
    params[pp(prefix, 'U')]  = np.concatenate([ortho_weight(dim),
                                               ortho_weight(dim)], axis=1)

    # embedding to hidden state proposal weights, biases
    # The followings appears in eq 8 where we compute the candidate h (tilde)
    params[pp(prefix, 'Wx')] = norm_weight(nin, dim, scale=scale)
    params[pp(prefix, 'bx')] = np.zeros((dim,)).astype(FLOAT)

    # recurrent transformation weights for hidden state proposal
    params[pp(prefix, 'Ux')] = ortho_weight(dim)

    if layernorm:
        params = init_layer_norm(prefix, params, dim)

    return params

def gru_layer(tparams, state_below, prefix='gru', mask=None, init_states=None,
              layernorm=False, truncate_grads=-1):
    nsteps = state_below.shape[0]

    # if we are dealing with a mini-batch
    n_samples = state_below.shape[1] if state_below.ndim == 3 else 1

    # Infer RNN dimensionality
    dim = tparams[pp(prefix, 'Ux')].shape[1]

    # if we have no mask, we assume all the inputs are valid
    if mask is None:
        # tensor.alloc(value, *shape)
        # mask: (n_steps, 1) filled with 1
        mask = tensor.alloc(1., nsteps, 1)

    # state_below is the input word embeddings
    # input to the gates, concatenated
    # [W_r * X + b_r, W_z * X + b_z]
    state_below_ = tensor.dot(state_below, tparams[pp(prefix, 'W')]) + tparams[pp(prefix, 'b')]

    # input to compute the hidden state proposal
    # This is the [W*x]_j in the eq. 8 of the paper
    state_belowx = tensor.dot(state_below, tparams[pp(prefix, 'Wx')]) + tparams[pp(prefix, 'bx')]

    # prepare scan arguments
    seqs = [mask, state_below_, state_belowx]
    if init_states is None:
        init_states = [tensor.alloc(0., n_samples, dim)]

    shared_vars = [tparams[pp(prefix, 'U')],
                   tparams[pp(prefix, 'Ux')]]

    _step = gru_step
    if layernorm:
        _step = gru_step_lnorm
        # bias and scale
        for i in ['b', 's']:
            # 4 for each
            for j in ['1','2','3','4']:
                shared_vars.append(tparams[pp(prefix, i+j)])

    rval, updates = theano.scan(_step,
                                sequences=seqs,
                                outputs_info=init_states,
                                non_sequences=shared_vars,
                                name=pp(prefix, '_layers'),
                                n_steps=nsteps,
                                truncate_gradient=truncate_grads,
                                strict=True)
    rval = [rval]
    return rval

######################################
# Conditional GRU layer with Attention
######################################
def param_init_gru_cond(params, nin, dim, dimctx, scale=0.01, prefix='gru_cond', layernorm=False):
    # nin:      input dim (e.g. embedding dim in the case of NMT)
    # dim:      gru_dim   (e.g. 1000)
    # dimctx:   2*gru_dim (e.g. 2000)

    # Parameters of the first GRU (+ lnorm params if requested)
    params = param_init_gru(params, nin, dim, scale, prefix, layernorm)

    # Below ones are new to this layer
    params[pp(prefix, 'U_nl')]          = np.concatenate([ortho_weight(dim),
                                                          ortho_weight(dim)], axis=1)
    params[pp(prefix, 'b_nl')]          = np.zeros((2 * dim,)).astype(FLOAT)

    params[pp(prefix, 'Ux_nl')]         = ortho_weight(dim)
    params[pp(prefix, 'bx_nl')]         = np.zeros((dim,)).astype(FLOAT)

    # context to GRU
    params[pp(prefix, 'Wc')]            = norm_weight(dimctx, dim*2, scale=scale)
    params[pp(prefix, 'Wcx')]           = norm_weight(dimctx, dim, scale=scale)

    ####### Attention
    # attention: combined -> hidden
    params[pp(prefix, 'W_comb_att')]    = norm_weight(dim, dimctx, scale=scale)

    # attention: context -> hidden
    # attention: hidden bias
    params[pp(prefix, 'Wc_att')]        = norm_weight(dimctx, dimctx, scale=scale)
    params[pp(prefix, 'b_att')]         = np.zeros((dimctx,)).astype(FLOAT)

    # attention: This gives the alpha's
    params[pp(prefix, 'U_att')]         = norm_weight(dimctx, 1, scale=scale)
    params[pp(prefix, 'c_att')]         = np.zeros((1,)).astype(FLOAT)

    return params

def gru_cond_layer(tparams, state_below, context, prefix='gru_cond',
                   mask=None, one_step=False, init_state=None, context_mask=None,
                   layernorm=False, truncate_grads=-1):
    if one_step:
        assert init_state, 'previous state must be provided'

    # Context
    # n_timesteps x n_samples x ctxdim
    assert context, 'Context must be provided'
    assert context.ndim == 3, 'Context must be 3-d: #annotation x #sample x dim'

    nsteps = state_below.shape[0]

    # Batch or single sample?
    n_samples = state_below.shape[1] if state_below.ndim == 3 else 1

    # if we have no mask, we assume all the inputs are valid
    # tensor.alloc(value, *shape)
    # mask: (n_steps, 1) filled with 1
    if mask is None:
        mask = tensor.alloc(1., nsteps, 1)

    # Infer RNN dimensionality
    dim = tparams[pp(prefix, 'Wcx')].shape[1]

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
    # Linearly transform the context to another space with same dimensionality
    pctx_ = tensor.dot(context, tparams[pp(prefix, 'Wc_att')]) + tparams[pp(prefix, 'b_att')]

    # Prepare for step()
    seqs = [mask, state_below_, state_belowx]
    shared_vars = [tparams[pp(prefix, 'U')],
                   tparams[pp(prefix, 'Wc')],
                   tparams[pp(prefix, 'W_comb_att')],
                   tparams[pp(prefix, 'U_att')],
                   tparams[pp(prefix, 'c_att')],
                   tparams[pp(prefix, 'Ux')],
                   tparams[pp(prefix, 'Wcx')],
                   tparams[pp(prefix, 'U_nl')],
                   tparams[pp(prefix, 'Ux_nl')],
                   tparams[pp(prefix, 'b_nl')],
                   tparams[pp(prefix, 'bx_nl')]]

    internal_step = gru_step
    if layernorm:
        internal_step = gru_step_lnorm
        # bias and scale
        for i in ['b', 's']:
            # 4 for each
            for j in ['1','2','3','4']:
                shared_vars.append(tparams[pp(prefix, i+j)])

    # Step function for the recurrence/scan
    # Sequences
    # ---------
    # m_    : mask
    # x_    : state_below_
    # xx_   : state_belowx
    # outputs_info
    # ------------
    # h_    : init_state,
    # ctx_  : need to be defined as it's returned by _step
    # alpha_: need to be defined as it's returned by _step
    # non sequences
    # -------------
    # pctx_ : pctx_
    # cc_   : context
    # and all the shared weights and biases..
    def _step(m_, x_, xx_,
              h_, ctx_, alpha_,
              pctx_, cc_, U, Wc, W_comb_att, U_att, c_att, Ux, Wcx, U_nl, Ux_nl, b_nl, bx_nl,
              *args):

        # Do a step of classical GRU
        h1 = internal_step(m_, x_, xx_, h_, U, Ux, *args)

        ###########
        # Attention
        ###########
        # h1 X W_comb_att
        # W_comb_att: dim -> dimctx
        # pstate_ should be 2D as we're working with unrolled timesteps
        pstate_ = tensor.dot(h1, W_comb_att)

        # Accumulate in pctx__ and apply tanh()
        # This becomes the projected context + the current hidden state
        # of the decoder, e.g. this is the information accumulating
        # into the returned original contexts with the knowledge of target
        # sentence decoding.
        pctx__ = tanh(pctx_ + pstate_[None, :, :])

        # Affine transformation for alpha = (pctx__ X U_att) + c_att
        # We're now down to scalar alpha's for each accumulated
        # context (0th dim) in the pctx__
        # alpha should be n_timesteps, n_samples, 1
        alpha = tensor.dot(pctx__, U_att) + c_att

        # Drop the last dimension, e.g. (n_timesteps, n_samples)
        alpha = alpha.reshape([alpha.shape[0], alpha.shape[1]])

        # Exponentiate alpha
        alpha = tensor.exp(alpha - alpha.max(0, keepdims=True))

        # If there is a context mask, multiply with it to cancel unnecessary steps
        if context_mask:
            alpha = alpha * context_mask

        # Normalize so that the sum makes 1
        alpha = alpha / alpha.sum(0, keepdims=True)

        # Compute the current context ctx_ as the alpha-weighted sum of
        # the initial contexts: context
        ctx_ = (cc_ * alpha[:, :, None]).sum(0)

        ###########################################
        # ctx_ and alpha computations are completed
        ###########################################

        ####################################
        # The below code is another GRU cell
        ####################################
        # Affine transformation: h1 X U_nl + b_nl
        # U_nl, b_nl: Stacked dim*2
        preact = tensor.dot(h1, U_nl) + b_nl

        # Transform the weighted context sum with Wc
        # and add it to preact
        # Wc: dimctx -> Stacked dim*2
        preact2 = preact + tensor.dot(ctx_, Wc)

        # Apply sigmoid nonlinearity
        preact2 = sigmoid(preact2)

        # Slice activations: New gates r2 and u2
        r2 = tensor_slice(preact2, 0, dim)
        u2 = tensor_slice(preact2, 1, dim)

        preactx = (tensor.dot(h1, Ux_nl) + bx_nl) * r2
        preactx2 = preactx + tensor.dot(ctx_, Wcx)

        # Candidate hidden
        h2_tilda = tanh(preactx2)

        # Leaky integration between the new h2 and the
        # old h1 computed in line 285
        h2 = u2 * h2_tilda + (1. - u2) * h1
        h2 = m_[:, None] * h2 + (1. - m_)[:, None] * h1

        return h2, ctx_, alpha.T

    if one_step:
        rval = _step(*(seqs + [init_state, None, None, pctx_, context] + shared_vars))
    else:
        outputs_info=[init_state,
                      tensor.alloc(0., n_samples, context.shape[2]), # ctxdim       (ctx_)
                      tensor.alloc(0., n_samples, context.shape[0])] # n_timesteps  (alpha)

        rval, updates = theano.scan(_step,
                                    sequences=seqs,
                                    outputs_info=outputs_info,
                                    non_sequences=[pctx_, context] + shared_vars,
                                    name=pp(prefix, '_layers'),
                                    n_steps=nsteps,
                                    truncate_gradient=truncate_grads,
                                    strict=True)
    return rval

#############################################
# LSTM
# https://github.com/kelvinxu/arctic-captions
#############################################
def param_init_lstm(params, nin, dim, forget_bias=0, scale=0.01, prefix='lstm'):
    """
     Stack the weight matrices for all the gates
     for much cleaner code and slightly faster dot-prods
    """
    # input weights
    # W_ix: Input x to input gate
    # W_fx: Input x to forget gate
    # W_ox: Input x to output gate
    # W_cx: Input x to cell state
    params[pp(prefix, 'W')] = np.concatenate([norm_weight(nin, dim, scale=scale),
                                              norm_weight(nin, dim, scale=scale),
                                              norm_weight(nin, dim, scale=scale),
                                              norm_weight(nin, dim, scale=scale)], axis=1)

    # for the previous hidden activation
    # W_im: Memory t_1 to input(t)
    # W_fm: Memory t_1 to forget(t)
    # W_om: Memory t_1 to output(t)
    # W_cm: Memory t_1 to cellstate(t)
    params[pp(prefix, 'U')] = np.concatenate([ortho_weight(dim),
                                              ortho_weight(dim),
                                              ortho_weight(dim),
                                              ortho_weight(dim)], axis=1)

    b = np.zeros((4 * dim,)).astype(FLOAT)
    b[dim: 2*dim] = forget_bias
    params[pp(prefix, 'b')] = b

    return params

# This function implements the lstm fprop
def lstm_layer(tparams, state_below, init_state=None, init_memory=None, one_step=False, prefix='lstm'):

    #if one_step:
    #    assert init_memory, 'previous memory must be provided'
    #    assert init_state, 'previous state must be provided'

    # number of timesteps
    nsteps = state_below.shape[0]

    # hidden dimension of LSTM layer
    dim = tparams[pp(prefix, 'U')].shape[0]

    if state_below.ndim == 3:
        # This is minibatch
        n_samples = state_below.shape[1]
    else:
        # during sampling, only single sample is received
        n_samples = 1

    if init_state is None:
        # init_state is dim per sample all zero
        init_state = tensor.alloc(0., n_samples, dim)

    if init_memory is None:
        # init_memory is dim per sample all zero
        init_memory = tensor.alloc(0., n_samples, dim)

    # This maps the input to LSTM dimensionality
    state_below = tensor.dot(state_below, tparams[pp(prefix, 'W')]) + tparams[pp(prefix, 'b')]

    ###########################
    # one time step of the lstm
    ###########################
    def _step(x_, m_, c_):
        """
           x_: state_below
           m_: init_memory
           c_: init_cell_state (this is actually not used when initializing)
        """
        
        preact = tensor.dot(m_, tparams[pp(prefix, 'U')])
        preact += x_

        # input(t) = sigm(W_ix * x_t + W_im * m_tm1)
        i = sigmoid(tensor_slice(preact, 0, dim))
        f = sigmoid(tensor_slice(preact, 1, dim))
        o = sigmoid(tensor_slice(preact, 2, dim))

        # cellstate(t)?
        c = tanh(tensor_slice(preact, 3, dim))

        # cellstate(t) = forget(t) * cellstate(t-1) + input(t) * cellstate(t)
        c = f * c_ + i * c

        # m_t, e.g. memory in tstep T in NIC paper
        m = o * tanh(c)

        return m, c

    if one_step:
        rval = _step(state_below, init_memory, init_state)
    else:
        rval, updates = theano.scan(_step,
                                    sequences=[state_below],
                                    outputs_info=[init_memory, init_state],
                                    name=pp(prefix, '_layers'),
                                    n_steps=nsteps)
    return rval
