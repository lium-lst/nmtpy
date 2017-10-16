# -*- coding: utf-8 -*-
import numpy as np
import pickle

from collections import OrderedDict
from .defaults import INT, FLOAT

def invert_dictionary(d):
    return OrderedDict([(v,k) for k,v in d.items()])

def load_dictionary(fname):
    with open(fname, 'rb') as f:
        vocab = pickle.load(f)

    return vocab, invert_dictionary(vocab)

# Function to convert idxs to sentence
def idx_to_sent(ivocab, idxs, join=True):
    sent = []
    for widx in idxs:
        if widx == 0:
            break
        sent.append(ivocab.get(widx, "<unk>"))
    if join:
        return " ".join(sent)
    else:
        return sent

# Function to convert sentence to idxs
def sent_to_idx(vocab, tokens, limit=0):
    idxs = []
    for word in tokens:
        # Get token, 1 if not available
        idx = vocab.get(word, 1)
        if limit > 0:
            idx = idx if idx < limit else 1
        idxs.append(idx)
    return idxs

# push parameters to Theano shared variables
def zipp(params, tparams):
    for kk, vv in params.items():
        tparams[kk].set_value(vv)

# pull parameters from Theano shared variables
def unzip(zipped):
    new_params = OrderedDict()
    for kk, vv in zipped.items():
        new_params[kk] = vv.get_value()
    return new_params

# make prefix-appended name
def pp(prefix, name):
    return '%s_%s' % (prefix, name)

# orthogonal initialization for weights
# Saxe, Andrew M., James L. McClelland, and Surya Ganguli.
# "Exact solutions to the nonlinear dynamics of learning in deep
# linear neural networks." arXiv preprint arXiv:1312.6120 (2013).
def ortho_weight(ndim):
    W = np.random.randn(ndim, ndim)
    u, s, v = np.linalg.svd(W)
    return u.astype(FLOAT)

# weight initializer, normal by default
def norm_weight(nin, nout, scale=0.01, ortho=True):
    if scale == "xavier":
        # Glorot, Xavier, and Yoshua Bengio. "Understanding the difficulty of training deep feedforward neural networks."
        # International conference on artificial intelligence and statistics. 2010.
        # http://machinelearning.wustl.edu/mlpapers/paper_files/AISTATS2010_GlorotB10.pdf
        scale = 1. / np.sqrt(nin)
    elif scale == "he":
        # Claimed necessary for ReLU
        # Kaiming He et al. (2015)
        # Delving deep into rectifiers: Surpassing human-level performance on
        # imagenet classification. arXiv preprint arXiv:1502.01852.
        scale = 1. / np.sqrt(nin/2.)

    if nout == nin and ortho:
        W = ortho_weight(nin)
    else:
        W = scale * np.random.randn(nin, nout)
    return W.astype(FLOAT)
