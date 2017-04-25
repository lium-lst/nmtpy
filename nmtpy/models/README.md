Models
------

Here we'll walkthrough over the multimodal CGRU implementation from `fusion_concat_dep_dep.py`.

```python
def gru_decoder_multi(tparams, state_below,
                      ctx1, ctx2, prefix='gru_decoder_multi',
                      input_mask=None, one_step=False,
                      init_state=None, ctx1_mask=None):
```
This is the function that creates the computational graph related to multimodal CGRU decoder. Let's look at the arguments first:
 - `tparams` is a dictionary containing all Theano variables of the model.
 - `state_below` is the input of the decoder which are the target embeddings in the classical NMT formulation.
 - `ctx1`: A tensor with shape `(max_seq_len, n_samples, 2*enc_hidden_dim)` which contains the set of hidden states produced by source encoder over the source sentence. The feature dimension is `2*enc_hidden_dim` because we concatenate forward and backward hidden states in the default bi-directional scheme.
 - `ctx2`: A tensor with shape `(w*h, n_samples, n_feature_maps)` which contains convolutional image features extracted from an arbitrary CNN. In order to apply visual attention over image zones, we flatten the spatial dimensions, i.e. an image is represented with a 2D matrix of shape `196x1024` instead of `14x14x1024` for ResNet-50 **res4f_relu** features. In this formulation, each row becomes a feature vector of `1024D` for a specific spatial location.
 - `prefix`: A prefix is appended to each network parameter name and here its purpose is to correctly fetch the parameters related to this computation block from `tparams`.
 - Since the data is organized into minibatches, we have `input_mask` and `ctx1_mask` which when multiplied with input and `ctx1` respectively, zeroes out padded positions for each sample. Since `ctx2` is the image features and image features are not of fixed size for each sample, we don't have a mask for it.
 - `init_state` defines the initial hidden state of the decoder.
 - `one_step` is a special mode where the computation graph will be executed step-by-step, necessary for translation decoding using beam-search.

```python
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
    dim = tparams[pp(prefix, 'Wcx')].shape[1]

    # initial/previous state
    # if not given, assume it's all zeros
    if init_state is None:
        init_state = tensor.alloc(0., n_samples, dim)
```
If `init_state` is not given, we assume the decoder is initialized with all zeros. Note that the state should be given during beam-search, i.e. when `one_step==True`.

```python
    # These two dot products are same with gru_layer, refer to the equations.
    # [W_r * X + b_r, W_z * X + b_z]
    state_below_ = tensor.dot(state_below, tparams[pp(prefix, 'W')]) + tparams[pp(prefix, 'b')]

    # input to compute the hidden state proposal
    # This is the [W*x]_j in the eq. 8 of the paper
    state_belowx = tensor.dot(state_below, tparams[pp(prefix, 'Wx')]) + tparams[pp(prefix, 'bx')]
```

Since this complex computation block encloses two consecutive GRUs, the boilerplate GRU calculations are embedded inside. So the above 2 transformations are the ones from a classical GRU layer, not specific to multimodality or `nmtpy`.

```python
    # Wc_att: dimctx -> dimctx
    # Linearly transform the contexts to another space with same dimensionality
    pctx1_ = tensor.dot(ctx1, tparams[pp(prefix, 'Wc_att')]) + tparams[pp(prefix, 'b_att')]
    pctx2_ = tensor.dot(ctx2, tparams[pp(prefix, 'Wc_att2')]) + tparams[pp(prefix, 'b_att2')]
```

We apply an affine transformation to both `ctx1` and `ctx2`.

```python
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
    # pctx2_ : pctx2_
    # cc1_   : ctx1
    # cc2_   : ctx2
    # and all the shared weights and biases..
    def _step(m_, x_, xx_,
              h_, ctx_, alpha1_, alpha2_, # These ctx and alpha's are not used in the computations
              pctx1_, pctx2_, cc1_, cc2_, U, Wc, W_comb_att, W_comb_att2,
              U_att, c_att, U_att2, c_att2,
              Ux, Wcx, U_nl, Ux_nl, b_nl, bx_nl, W_fus, c_fus):
 ```
    
`_step` is the function that will be executed by Theano's `scan` mechanism, i.e. this is the core recurrent computation unit of the decoder RNN. The first three arguments `m_, x_, xx_` are the sequence arguments which mean that `scan()` will iterate over their first axis and call `_step()` on it. `outputs_info` are the tensors which are recurrently fed back to each new `_step()` call. Even if you do not use a tensor's previous value in the new timestep, you need to provide it if you want to collect them at the end of the recurrence. Finally what's left are the other tensors mostly weights and biases used during recurrence.

        # Do a step of classical GRU
        h1 = gru_step(m_, x_, xx_, h_, U, Ux)

        ############################################
        # NOTE: Distinct attention for each modality
        ############################################
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
        pctx2__ = tanh(pctx2_ + pstate2_[None, :, :])

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
        ctx1_ = (cc1_ * alpha1[:, :, None]).sum(0)
        ctx2_ = (cc2_ * alpha2[:, :, None]).sum(0)
        # n_samples x ctxdim (2000)

        ##############################################
        # NOTE: This is the fusion context with concat
        ##############################################
        ctx_ = tensor.dot(tensor.concatenate([ctx1_, ctx2_], axis=1), W_fus) + c_fus

        ############################################
        # ctx*_ and alpha computations are completed
        ############################################

        ####################################
        # The below code is another GRU cell
        ####################################
        # Affine transformation: h1 X U_nl + b_nl
        # U_nl, b_nl: Stacked dim*2
        preact = tensor.dot(h1, U_nl) + b_nl

        # Transform the weighted context sum with Wc
        # and add it to preact
        # Wc: dimctx -> Stacked dim*2
        preact += tensor.dot(ctx_, Wc)

        # Apply sigmoid nonlinearity
        preact = sigmoid(preact)

        # Slice activations: New gates r2 and u2
        r2 = tensor_slice(preact, 0, dim)
        u2 = tensor_slice(preact, 1, dim)

        preactx = (tensor.dot(h1, Ux_nl) + bx_nl) * r2
        preactx += tensor.dot(ctx_, Wcx)

        # Candidate hidden
        h2_tilda = tanh(preactx)

        # Leaky integration between the new h2 and the
        # old h1 computed in line 285
        h2 = u2 * h2_tilda + (1. - u2) * h1
        h2 = m_[:, None] * h2 + (1. - m_)[:, None] * h1

        return h2, ctx_, alpha1.T, alpha2.T
        ```
