from six.moves import range
from six.moves import zip

# Python
import os
import cPickle
import inspect
import importlib
import copy

from collections import OrderedDict

# 3rd party
import numpy as np

import theano
import theano.tensor as tensor

# Ours
from ..layers import *
from ..typedef import *
from ..nmtutils import *
from ..iterators.text import TextIterator
from ..iterators.bitext import BiTextIterator
from ..iterators.factorstext import FactorsIterator
from ..models.basemodel import BaseModel

from ..sysutils import *

class Model(BaseModel):
    def __init__(self, seed, **kwargs):
        # Call parent's init first
        super(Model, self).__init__(**kwargs)

        # Load dictionaries
        dicts = kwargs['dicts']
        self.src_dict, src_idict = load_dictionary(dicts['src'])
        self.n_words_src = min(self.n_words_src, len(self.src_dict)) \
                if self.n_words_src > 0 else len(self.src_dict)
        self.trg_dict, trg_idict = load_dictionary(dicts['trg'])
        self.n_words_trg = min(self.n_words_trg, len(self.trg_dict)) \
                if self.n_words_trg > 0 else len(self.trg_dict)
        self.trgmult_dict, trgmult_idict = load_dictionary(dicts['trgmult'])
        self.n_words_trgmult = min(self.n_words_trgmult, len(self.trgmult_dict)) \
                if self.n_words_trgmult > 0 else len(self.trgmult_dict)

        # Use GRU by default as encoder
        self.enc_type = kwargs.get('enc_type', 'gru')

	# Should we normalize train cost or not?
	self.norm_cost = kwargs.get('norm_cost', False)
        
	# Shuffle sentences by target length or not
	self.shuffle_mode = kwargs.get('shuffle_mode', None)
	
	# Use a single embedding matrix for target words
        self.tied_trg_emb = kwargs.get('tied_trg_emb', False)

        # Create options. This will saved as .pkl
        self.set_options(self.__dict__)

        self.src_idict = src_idict
        self.trg_idict = trg_idict
        self.trgmult_idict = trgmult_idict

        self.ctx_dim = 2 * self.rnn_dim
        self.set_trng(seed)

        # We call this once to setup dropout mechanism correctly
        self.set_dropout(False)
	# TODO charge lookup table (it should be in run_beam_search and pass it to sysutils.py)
	self.dic_lem = dict()
	#self.lookup_word_fact = '/lium/buster1/garcia/workspace/scripts/lookup_table_ar.txt'
	self.lookup_word_fact = '/lium/buster1/garcia/workspace/scripts/dico_french_clean.txt'
	lookup_word_fact = open(self.lookup_word_fact)
	# build a def in the future
	#def build_dic(table): 
	# Parse the table
	for line in lookup_word_fact:
	    values = line.strip().split(" ")
		 
	    if values[0][0] != '#':
	        #wo = values[1][3:]
	        po = values[2][3:]
	        le = values[3][3:]
	        te = values[4][3:]
	        pe = values[5][3:]
	        ge = values[6][3:]
	        nu = values[7][3:]
	        if self.trg_dict.get(le, None) != None: 
		    le_id = self.trg_dict.get(le)
		    self.dic_lem[le_id]=self.dic_lem.get(le_id,[])
		    for ca in 'u', 'l', 'c':
		        fa = "%s|%s|%s|%s|%s|%s"%(po, te, pe, ge, nu, ca)
		        fa_id = self.trgmult_dict.get(fa)
		        if fa_id != None and fa_id not in self.dic_lem[le_id]:
			   self.dic_lem[le_id].append(fa_id)

#	    for line in lookup_word_fact:
#		wo, le, fa = line.strip().split('|')
#		# They key of the dic is the lemma and factors are the values
#		key = "%s"%(le)
#		dic_lem[key] = dic_lem.get(key,[])
#		if fa not in dic_lem[key]:
#		    # add possible factors for each lemma
#		    dic_lem[key].append(fa)
	    #return dic_lem
	print ('lookuptable loaded')
	print ('self.n_words_trgmult=', self.n_words_trgmult)



    def info(self, logger):
        logger.info('Source vocabulary size: %d', self.n_words_src)
        logger.info('Target vocabulary size: %d', self.n_words_trg)
        logger.info('Target mult vocabulary size: %d', self.n_words_trgmult)
        logger.info('%d training samples' % self.train_iterator.n_samples)
        logger.info('%d validation samples' % self.valid_iterator.n_samples)

    def load_valid_data(self, from_translate=False):
        self.valid_ref_files = self.data['valid_trg']
        if isinstance(self.valid_ref_files, str):
            self.valid_ref_files = list([self.valid_ref_files])

        if from_translate:
            self.valid_iterator = TextIterator(
                                    batch_size=1,
                                    file=self.data['valid_src'], dict=self.src_dict,
                                    n_words=self.n_words_src)
        else:
            # Take the first validation item for NLL computation
            self.valid_iterator = FactorsIterator(
                                    batch_size=self.batch_size,
                                    srcfile=self.data['valid_src'], srcdict=self.src_dict,
                                    trglemfile=self.data['valid_trglem'], trglemdict=self.trg_dict,
                                    trgfactfile=self.data['valid_trgfact'], trgfactdict=self.trgmult_dict,
                                    n_words_src=self.n_words_src, 
				    n_words_trglem=self.n_words_trg, n_words_trgfact=self.n_words_trgmult)

        self.valid_iterator.read()

    def load_data(self):
        
	self.train_iterator = FactorsIterator(
                                batch_size=self.batch_size,
                                shuffle_mode=self.shuffle_mode,
                                srcfile=self.data['train_src'], srcdict=self.src_dict,
                                trglemfile=self.data['train_trg'], trglemdict=self.trg_dict,
                                trgfactfile=self.data['train_trgmult'], trgfactdict=self.trgmult_dict,
                                n_words_src=self.n_words_src,
                                n_words_trglem=self.n_words_trg, n_words_trgfact=self.n_words_trgmult)

        # Prepare batches
        self.train_iterator.read()
        self.load_valid_data()

    def init_params(self):
        params = OrderedDict()

        # embedding weights for encoder and decoder
        params['Wemb_enc'] = norm_weight(self.n_words_src, self.embedding_dim, scale=self.weight_init)
        params['Wemb_dec_lem'] = norm_weight(self.n_words_trg, self.embedding_dim, scale=self.weight_init)
	params['Wemb_dec_fact'] = norm_weight(self.n_words_trgmult, self.embedding_dim, scale=self.weight_init)

        # encoder: bidirectional RNN
        #########
        # Forward encoder
        params = get_new_layer(self.enc_type)[0](params, prefix='encoder', nin=self.embedding_dim, dim=self.rnn_dim, scale=self.weight_init)
        # Backwards encoder
        params = get_new_layer(self.enc_type)[0](params, prefix='encoder_r', nin=self.embedding_dim, dim=self.rnn_dim, scale=self.weight_init)

        # Context is the concatenation of forward and backwards encoder

        # init_state, init_cell
        params = get_new_layer('ff')[0](params, prefix='ff_state', nin=self.ctx_dim, nout=self.rnn_dim, scale=self.weight_init)
        # decoder
        params = get_new_layer('gru_cond')[0](params, prefix='decoder', nin=2*self.embedding_dim, dim=self.rnn_dim, dimctx=self.ctx_dim, scale=self.weight_init)

        # readout
        params = get_new_layer('ff')[0](params, prefix='ff_logit_gru'  , nin=self.rnn_dim       , nout=self.embedding_dim, scale=self.weight_init, ortho=False)
        params = get_new_layer('ff')[0](params, prefix='ff_logit_prev' , nin=2*self.embedding_dim , nout=self.embedding_dim, scale=self.weight_init, ortho=False)
        params = get_new_layer('ff')[0](params, prefix='ff_logit_ctx'  , nin=self.ctx_dim       , nout=self.embedding_dim, scale=self.weight_init, ortho=False)
	if self.tied_trg_emb is False:
	    params = get_new_layer('ff')[0](params, prefix='ff_logit_trg'      , nin=self.embedding_dim , nout=self.n_words_trg, scale=self.weight_init)
	    params = get_new_layer('ff')[0](params, prefix='ff_logit_trgmult'      , nin=self.embedding_dim , nout=self.n_words_trgmult, scale=self.weight_init)

        self.initial_params = params

    def build(self):
        # description string: #words x #samples
        x = tensor.matrix('x', dtype=INT)
        x_mask = tensor.matrix('x_mask', dtype=FLOAT)
        y1 = tensor.matrix('y1', dtype=INT)
        y1_mask = tensor.matrix('y1_mask', dtype=FLOAT)
        y2 = tensor.matrix('y2', dtype=INT)

        self.inputs['x'] = x
        self.inputs['x_mask'] = x_mask
        self.inputs['y1'] = y1
        self.inputs['y2'] = y2
        self.inputs['y1_mask'] = y1_mask

        # for the backward rnn, we just need to invert x and x_mask
        xr = x[::-1]
        xr_mask = x_mask[::-1]

        n_timesteps = x.shape[0]
        n_timesteps_trg = y1.shape[0]
        n_timesteps_trgmult = y2.shape[0]
        n_samples = x.shape[1]

        # word embedding for forward rnn (source)
        emb = self.tparams['Wemb_enc'][x.flatten()]
        emb = emb.reshape([n_timesteps, n_samples, self.embedding_dim])
        proj = get_new_layer(self.enc_type)[1](self.tparams, emb, prefix='encoder', mask=x_mask)

        # word embedding for backward rnn (source)
        embr = self.tparams['Wemb_enc'][xr.flatten()]
        embr = embr.reshape([n_timesteps, n_samples, self.embedding_dim])
        projr = get_new_layer(self.enc_type)[1](self.tparams, embr, prefix='encoder_r', mask=xr_mask)

        # context will be the concatenation of forward and backward rnns
        ctx = tensor.concatenate([proj[0], projr[0][::-1]], axis=proj[0].ndim-1)

        # mean of the context (across time) will be used to initialize decoder rnn
        ctx_mean = (ctx * x_mask[:, :, None]).sum(0) / x_mask.sum(0)[:, None]

        # initial decoder state
        init_state = get_new_layer('ff')[1](self.tparams, ctx_mean, prefix='ff_state', activ='tanh')

        # word embedding (target), we will shift the target sequence one time step
        # to the right. This is done because of the bi-gram connections in the
        # readout and decoder rnn. The first target will be all zeros and we will
        # not condition on the last output.
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
	emb_prev = tensor.concatenate([emb_lem,emb_fact], axis=2)

        # decoder - pass through the decoder conditional gru with attention
        proj = get_new_layer('gru_cond')[1](self.tparams, emb_prev,
                                            prefix='decoder',
                                            mask=y1_mask, context=ctx,
                                            context_mask=x_mask,
                                            one_step=False,
                                            init_state=init_state)
        # hidden states of the decoder gru
        proj_h = proj[0]

        # weighted averages of context, generated by attention module
        ctxs = proj[1]

        # weights (alignment matrix)
        alphas = proj[2]

        # compute word probabilities
        logit_gru  = get_new_layer('ff')[1](self.tparams, proj_h, prefix='ff_logit_gru', activ='linear')
        logit_ctx  = get_new_layer('ff')[1](self.tparams, ctxs, prefix='ff_logit_ctx', activ='linear')
        logit_prev = get_new_layer('ff')[1](self.tparams, emb_prev, prefix='ff_logit_prev', activ='linear')

        logit = tanh(logit_gru + logit_prev + logit_ctx)

        if self.dropout > 0:
            logit = dropout_layer(logit, self.use_dropout, self.dropout, self.trng)

	if self.tied_trg_emb is False:
	    logit_trg = get_new_layer('ff')[1](self.tparams, logit, prefix='ff_logit_trg', activ='linear')
	    logit_trgmult = get_new_layer('ff')[1](self.tparams, logit, prefix='ff_logit_trgmult', activ='linear')
	else:
	    logit_trg = tensor.dot(logit, self.tparams['Wemb_dec_lem'].T)
	    logit_trgmult = tensor.dot(logit, self.tparams['Wemb_dec_fact'].T)

        logit_trg_shp = logit_trg.shape
        logit_trgmult_shp = logit_trgmult.shape

        # Apply logsoftmax (stable version)
        log_trg_probs = -tensor.nnet.logsoftmax(logit_trg.reshape([logit_trg_shp[0]*logit_trg_shp[1], logit_trg_shp[2]]))
        log_trgmult_probs = -tensor.nnet.logsoftmax(logit_trgmult.reshape([logit_trgmult_shp[0]*logit_trgmult_shp[1], logit_trgmult_shp[2]]))

        # cost
        y1_flat = y1.flatten()
        y2_flat = y2.flatten()
        y1_flat_idx = tensor.arange(y1_flat.shape[0]) * self.n_words_trg + y1_flat
        y2_flat_idx = tensor.arange(y2_flat.shape[0]) * self.n_words_trgmult + y2_flat

        cost_trg = log_trg_probs.flatten()[y1_flat_idx]
        cost_trg = cost_trg.reshape([y1.shape[0], y1.shape[1]])
        cost_trg = (cost_trg * y1_mask).sum(0)

        cost_trgmult = log_trgmult_probs.flatten()[y2_flat_idx]
        cost_trgmult = cost_trgmult.reshape([y2.shape[0], y2.shape[1]])
        cost_trgmult = (cost_trgmult * y1_mask).sum(0)

	cost = cost_trg + cost_trgmult
        self.f_log_probs = theano.function(self.inputs.values(), cost)

        # For alpha regularization
        self.x_mask = x_mask
        self.y1_mask = y1_mask
        self.alphas = alphas

        if self.norm_cost:
	    return (cost / y1_mask.sum(0)).mean()

        return cost.mean()

    def add_alpha_regularizer(self, cost, alpha_c):
        alpha_c = theano.shared(np.float32(alpha_c), name='alpha_c')
        alpha_reg = alpha_c * (
            (tensor.cast(self.y1_mask.sum(0) // self.x_mask.sum(0), FLOAT)[:, None] -
             self.alphas.sum(0))**2).sum(1).mean()
        cost += alpha_reg
        return cost

    def build_sampler(self):
        x = tensor.matrix('x', dtype=INT)
        xr = x[::-1]
        n_timesteps = x.shape[0]
        n_samples = x.shape[1]

        # word embedding (source), forward and backward
        emb = self.tparams['Wemb_enc'][x.flatten()]
        emb = emb.reshape([n_timesteps, n_samples, self.embedding_dim])

        embr = self.tparams['Wemb_enc'][xr.flatten()]
        embr = embr.reshape([n_timesteps, n_samples, self.embedding_dim])

        # encoder
        proj = get_new_layer(self.enc_type)[1](self.tparams, emb, prefix='encoder')
        projr = get_new_layer(self.enc_type)[1](self.tparams, embr, prefix='encoder_r')

        # concatenate forward and backward rnn hidden states
        ctx = tensor.concatenate([proj[0], projr[0][::-1]], axis=proj[0].ndim-1)

        # get the input for decoder rnn initializer mlp
        ctx_mean = ctx.mean(0)
        # ctx_mean = tensor.concatenate([proj[0][-1],projr[0][-1]], axis=proj[0].ndim-2)
        init_state = get_new_layer('ff')[1](self.tparams, ctx_mean, prefix='ff_state', activ='tanh')

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
                                         init_state=init_state)

        next_state = r[0]
        ctxs = r[1]
        alphas = r[2]

        logit_prev = get_new_layer('ff')[1](self.tparams, emb_prev,          prefix='ff_logit_prev',activ='linear')
        logit_ctx  = get_new_layer('ff')[1](self.tparams, ctxs,         prefix='ff_logit_ctx', activ='linear')
        logit_gru  = get_new_layer('ff')[1](self.tparams, next_state,   prefix='ff_logit_gru', activ='linear')

        logit = tanh(logit_gru + logit_prev + logit_ctx)

        if self.dropout > 0:
            logit = dropout_layer(logit, self.use_dropout, self.dropout, self.trng)

	if self.tied_trg_emb is False:
	    logit_trg = get_new_layer('ff')[1](self.tparams, logit, prefix='ff_logit_trg', activ='linear')
	    logit_trgmult = get_new_layer('ff')[1](self.tparams, logit, prefix='ff_logit_trgmult', activ='linear')
	else:
	    logit_trg = tensor.dot(logit, self.tparams['Wemb_dec_lem'].T)
	    logit_trgmult = tensor.dot(logit, self.tparams['Wemb_dec_fact'].T)

        # compute the logsoftmax
        next_log_probs_trg = tensor.nnet.logsoftmax(logit_trg)
        next_log_probs_trgmult = tensor.nnet.logsoftmax(logit_trgmult)

        # Sample from the softmax distribution
        next_probs_trg = tensor.exp(next_log_probs_trg)
        next_probs_trgmult = tensor.exp(next_log_probs_trgmult)
        next_word_trg = self.trng.multinomial(pvals=next_probs_trg).argmax(1)
        next_word_trgmult = self.trng.multinomial(pvals=next_probs_trgmult).argmax(1)

        # compile a function to do the whole thing above
        # next hidden state to be used
        inputs = [y1, y2, ctx, init_state]
        outs = [next_log_probs_trg, next_word_trg, next_log_probs_trgmult, next_word_trgmult, next_state, alphas]
        self.f_next = theano.function(inputs, outs, name='f_next')
    
    def run_beam_search(self, beam_size=12, n_jobs=8, metric='bleu', mode='beamsearch', out_file=None):
        # Save model temporarily
        with get_temp_file(suffix=".npz", delete=True) as tmpf:
            self.save(tmpf.name)

            result = get_valid_evaluation(tmpf.name,
                                          pkl_path=self.model_path + ".pkl",
                                          beam_size=beam_size,
                                          n_jobs=n_jobs,
                                          metric=metric,
                                          mode=mode,
                                          out_file=out_file,
					  factors=True)

        return result
    '''    
    def beam_search(self, inputs, beam_size=12, maxlen=50, suppress_unks=False, **kwargs):

	    # Final results and their scores
	    final_sample_lem = []
	    final_score_lem = []
	    final_sample_fact = []
	    final_score_fact = []
	    final_alignments = []

	    # Initially we have one empty hypothesis with a score of 0
	    hyp_scores  = np.zeros(1).astype(FLOAT)
	    hyp_scores_lem  = np.zeros(1).astype(FLOAT)
	    hyp_scores_fact  = np.zeros(1).astype(FLOAT)
	    hyp_samples_lem = [[]]
	    hyp_samples_fact = [[]]
	    hyp_alignments  = [[]]

	    # get initial state of decoder rnn and encoder context vectors
	    # ctx0: the set of context vectors leading to the next_state
	    # with a shape of (n_words x 1 x ctx_dim)
	    # next_state: mean context vector (ctx0.mean()) passed through FF with a final
	    # shape of (1 x 1 x ctx_dim)
	    # merc - The 2 outputs have the same next_state
	    next_state, ctx0 = self.f_init(inputs[0])

	    # Beginning-of-sentence indicator is -1
	    next_w_lem = -1 * np.ones((1,)).astype(INT)
	    next_w_fact = -1 * np.ones((1,)).astype(INT)

	    #print ('***********************************************')
	    # maxlen or 3 times source length
	    #maxlen = min(maxlen, inputs[0].shape[0] * 3)
	    maxlen = inputs[0].shape[0] * 3
	    #print ('maxlen=', maxlen)

            # Always starts with the initial tstep's context vectors
            # e.g. we have a ctx0 of shape (n_words x 1 x ctx_dim)
            # Tiling it live_beam times makes it (n_words x live_beam x ctx_dim)
            # thus we create sth like a batch of live_beam size with every word duplicated
            # for further state expansion.
            tiled_ctx = np.tile(ctx0, [1, 1])
            live_beam = beam_size

	    #src_sent = idx_to_sent(self.src_idict, inputs[0].flatten())
	    #print ('src_sent=', src_sent, len(src_sent.split(' ')))
	    


	    for ii in xrange(maxlen):
		#print ('---------------------------------------------------')
		#print ('ii=',ii)
		# Always starts with the initial tstep's context vectors
		# e.g. we have a ctx0 of shape (n_words x 1 x ctx_dim)
		# Tiling it live_beam times makes it (n_words x live_beam x ctx_dim)
		# thus we create sth like a batch of live_beam size with every word duplicated
		# for further state expansion.
		#print ('live_beam=', live_beam)
		#print ('next_w_lem=', next_w_lem)
		#print ('next_w_fact=', next_w_fact)

		# Get next states
		# In the first iteration, we provide -1 and obtain the log_p's for the
		# first word. In the following iterations tiled_ctx becomes a batch
		# of duplicated left hypotheses. tiled_ctx is always the same except
		# the 2nd dimension as the context vectors of the source sequence
		# is always the same regardless of the decoding step.
		next_log_p_lem, _, next_log_p_fact, _, next_state, alphas = self.f_next(*[next_w_lem, next_w_fact, tiled_ctx, next_state])
                # For each f_next, we obtain a new set of alpha's for the next_w
                # for each hypothesis in the beam search

                if suppress_unks:
                    next_log_p_lem[:, 1] = -np.inf


		# Compute sum of log_p's for the current n-gram hypotheses and flatten them
		cand_scores_lem = hyp_scores_lem[:, None] - next_log_p_lem
		cand_scores_fact = hyp_scores_fact[:, None] - next_log_p_fact
		
		# Beam search improvement for factors
		# Do combination for each new hyp
		cand_costs = []
		cand_costs_lem = []
		cand_costs_fact = []
		cand_w_idx = []
		cand_trans_idx = []
		for idx, [cand_h_scores_lem, cand_h_scores_fact] in enumerate(zip(cand_scores_lem, cand_scores_fact)):
		    # Take the best beam_size-dead_beam hypotheses
		    #ranks_lem = cand_h_scores_lem.argpartition(live_beam-1)[:live_beam]
		    ranks_lem = cand_h_scores_lem.argsort()[:(live_beam)]
		    # if beam size is bigger than factors vocab, use vocab size as beam size for it
		    if live_beam > self.n_words_trgmult:
			ranks_fact = cand_h_scores_fact.argsort()[:(self.n_words_trgmult)]
		    else:
			#ranks_fact = cand_h_scores_fact.argpartition(live_beam-1)[:live_beam]
			# we have to use argsort to order them, argpartition extracts the best beam size but not ordered
			ranks_fact = cand_h_scores_fact.argsort()[:(live_beam)]
			print ('ranks_fact:')
			print (ranks_fact.shape)
			print (ranks_fact)
			# keep all the hyps for factors, for the case that the previous are wrong
			#ranks_fact_all = cand_h_scores_fact.argpartition(self.n_words_trgmult-1)[:self.n_words_trgmult]
			ranks_fact_all = cand_h_scores_fact.argsort()[:(self.n_words_trgmult)]
			print ('ranks_fact_all:')
			print (ranks_fact_all.shape)
			print (ranks_fact_all)
			print ('ranks_fact2:')
			print (ranks_fact.shape)
			print (ranks_fact)
		    #print ('ranks_lem=', ranks_lem)
		    #print ('ranks_fact=', ranks_fact)
		    # Get their costs
		    costs_h_lem = cand_h_scores_lem[ranks_lem]
		    costs_h_fact = cand_h_scores_fact[ranks_fact]
		    print ('costs_h_fact:')
		    print (costs_h_fact.shape)
		    print (len(costs_h_fact))
		    costs_h_fact_all = cand_h_scores_fact[ranks_fact_all]
		    print ('costs_h_fact_all:')
		    print (costs_h_fact_all.shape)
		    print (len(costs_h_fact_all))
		    word_indices_lem = ranks_lem % self.n_words_trg
#		    print ('word_indices_lem=',idx_to_sent(self.trg_idict, word_indices_lem), len(word_indices_lem))
		    word_indices_fact = ranks_fact % self.n_words_trgmult
		    word_indices_fact_all = ranks_fact_all % self.n_words_trgmult
		    print ('word_indices_fact=',idx_to_sent(self.trgmult_idict, word_indices_fact), len(word_indices_fact))
		    print (word_indices_fact.shape)
		    print (word_indices_fact)
		    print ('word_indices_fact_all=',idx_to_sent(self.trgmult_idict, word_indices_fact_all), len(word_indices_fact_all))
		    print (word_indices_fact_all.shape)
		    print (word_indices_fact_all)
		    
		    # Sum the logp's of lemmas and factors and keep the best ones
		    cand_h_costs = []
		    cand_h_costs_lem = []
		    cand_h_costs_fact = []
		    cand_h_w_idx = []
		    for l in range(live_beam):
			lem = self.trg_idict.get(word_indices_lem[l])
			print ('lem=', lem, word_indices_lem[l])
			if live_beam > self.n_words_trgmult:
			    beam = self.n_words_trgmult
			else:
			    beam = live_beam
			fact_all = beam
			#TODO get the indexes of the possible factors
			for f in range(beam):
			    found = False
			    # Check if this combination of lemma and factors is correct
			    fact = self.trgmult_idict.get(word_indices_fact[f])
			    print ('Another fact=', fact)
			    # Check if we have the lemma in the table
			    if word_indices_lem[l] in self.dic_lem.keys():
				print ('lemma in the dic')
				correct_fact = self.dic_lem[word_indices_lem[l]]
				print ('word_indices_fact=',idx_to_sent(self.trgmult_idict, correct_fact), len(correct_fact))
				print ('cost of the correct:', costs_h_fact_all[correct_fact])
			        # if we have it, check if the factos are correct for this lemma
				if word_indices_fact[f] not in self.dic_lem[word_indices_lem[l]]:
				    print ('fact_incorrect=', fact, 'fact_all', fact_all)
				    # check if we have already seen all the possibilities
				    if fact_all < self.n_words_trgmult - 1:
					# Go to the next factors according to the probs
####					cand_h_costs.append(costs_h_lem[l]+ costs_h_fact[f])
					print ('COST1:', costs_h_lem[l]+ costs_h_fact[f])
####					cand_h_costs_lem.append(costs_h_lem[l])
####					cand_h_costs_fact.append(costs_h_fact[f])
					# keep the word indexes of both outputs
##				    	cand_h_w_idx.append([word_indices_lem[l], word_indices_fact[f]])
					for fa in range(fact_all,self.n_words_trgmult):
					    fact = self.trgmult_idict.get(word_indices_fact_all[fa])
					    print ('fact2=', fact, fa)
					    if found == False and word_indices_fact_all[fa] in self.dic_lem[word_indices_lem[l]]:
						print ('FOUND: lem=',lem,' fact_correct=', fact)
						print ('COST2:', costs_h_lem[l]+ costs_h_fact_all[fa])
						# Update the candidates with the correct factors
###					    	cand_h_costs.append(costs_h_lem[l]+ costs_h_fact_all[fa])
					    	cand_h_costs.append(costs_h_lem[l])
##					    	cand_h_costs.append(costs_h_lem[l]+ costs_h_fact[f])
##					    	cand_h_costs.append(costs_h_lem[l]+ costs_h_fact[f])
					    	cand_h_costs_lem.append(costs_h_lem[l])
					    	cand_h_costs_fact.append(costs_h_fact_all[fa])
##					    	cand_h_costs_fact.append(costs_h_fact[f])
						# keep the word indexes of both outputs
					    	cand_h_w_idx.append([word_indices_lem[l], word_indices_fact_all[fa]])
##					    	cand_h_w_idx.append([word_indices_lem[l], word_indices_fact[f]])
						# no to take the same factors as before
						fact_all = fa+1
						found = True
						#break
					if found == False:
					    print ('No more factors cause ending')
					    #cand_h_costs.append(costs_h_lem[l]+ costs_h_fact[f])
					    cand_h_costs.append(costs_h_lem[l])
					    cand_h_costs_lem.append(costs_h_lem[l])
					    cand_h_costs_fact.append(costs_h_fact[f])
					    # keep the word indexes of both outputs
					    cand_h_w_idx.append([word_indices_lem[l], word_indices_fact[f]])
				    else:
					print ('No more factors')
					#cand_h_costs.append(costs_h_lem[l]+ costs_h_fact[f])
					cand_h_costs.append(costs_h_lem[l])
					cand_h_costs_lem.append(costs_h_lem[l])
					cand_h_costs_fact.append(costs_h_fact[f])
					# keep the word indexes of both outputs
					cand_h_w_idx.append([word_indices_lem[l], word_indices_fact_all[fa]])
				else:
				    print ('NO PROB: lemma=', lem, 'and factors=', fact, 'are correct')
				    #cand_h_costs.append(costs_h_lem[l]+ costs_h_fact[f])
				    cand_h_costs.append(costs_h_lem[l])
				    cand_h_costs_lem.append(costs_h_lem[l])
				    cand_h_costs_fact.append(costs_h_fact[f])
				    # keep the word indexes of both outputs
				    cand_h_w_idx.append([word_indices_lem[l], word_indices_fact[f]])
			    else:
				# This is EOS or UNK, it does not matter the corresponding factors
				# TODO in this case the beam will be just the hyps coming from lemmas and the best factors hyp
				print ('WARNING:,'+str(l)+' '+lem.decode('utf-8').encode('utf-8')+' lemma not in the lookup table')
				#cand_h_costs.append(costs_h_lem[l]+ costs_h_fact[f])
				cand_h_costs.append(costs_h_lem[l])
				cand_h_costs_lem.append(costs_h_lem[l])
				cand_h_costs_fact.append(costs_h_fact[f])
				# keep the word indexes of both outputs
				cand_h_w_idx.append([word_indices_lem[l], word_indices_fact[f]])
		    # We convert the merged lists to np arrays and prune with the best costs and get indices of the nbest
		    cand_h_costs = np.array(cand_h_costs)
		    cand_h_costs_lem = np.array(cand_h_costs_lem)
		    cand_h_costs_fact = np.array(cand_h_costs_fact)
		    cand_h_w_idx = np.array(cand_h_w_idx)
		    ranks_h_costs = cand_h_costs.argsort()[:(live_beam)]
		    
		    # We append the beam_size hyps
		    cand_costs.append(cand_h_costs[ranks_h_costs])
		    cand_costs_lem.append(cand_h_costs_lem[ranks_h_costs])
		    cand_costs_fact.append(cand_h_costs_fact[ranks_h_costs])
		    word_h_indices = cand_h_w_idx[ranks_h_costs]
		    # We cannot flatten later this array, we need pair elements
		    for w in word_h_indices:
			cand_w_idx.append(w)
		    trans_h_indices = []
		    trans_h_indices = beam_size * [idx]
		    trans_h_indices = np.array(trans_h_indices)
		    cand_trans_idx.append(trans_h_indices)

		# We convert the merged lists to np arrays and prune with the best costs and get indices of the nbest
		cand_costs = np.array(cand_costs)
		cand_costs_lem = np.array(cand_costs_lem)
		cand_costs_fact = np.array(cand_costs_fact)
		cand_w_idx = np.array(cand_w_idx)
		cand_trans_idx = np.array(cand_trans_idx)
		print ('cand_costs.shape=', cand_costs.shape)
		cand_flat_costs = cand_costs.flatten()
		cand_flat_costs_lem = cand_costs_lem.flatten()
		cand_flat_costs_fact = cand_costs_fact.flatten()
		print ('live_beam=', live_beam)
		print ('cand_flat_costs.shape=', cand_flat_costs.shape)
		ranks_costs = cand_flat_costs.argsort()[:(live_beam)]
		costs = cand_flat_costs[ranks_costs]
		costs_lem = cand_flat_costs_lem[ranks_costs]
		costs_fact = cand_flat_costs_fact[ranks_costs]
		word_indices = cand_w_idx[ranks_costs]
		cand_trans_idx_flat = cand_trans_idx.flatten()
		trans_indices = cand_trans_idx_flat[ranks_costs]

		#print ('#########################################################')

		# New states, scores and samples
		live_beam = 0
		# We have shared scores for both outputs after the last pruning
		new_hyp_scores = []
		new_hyp_samples_lem = []
		new_hyp_scores_lem = []
		new_hyp_samples_fact = []
		# Using the EOS of lemmas for factors
		new_hyp_scores_fact = []
		new_hyp_alignments  = []

		# This will be the new next states in the next iteration
	        hyp_states = []

		# Iterate over the hypotheses and add them to new_* lists
		# We have common next_state
		for idx, [wi, ti] in enumerate(zip(word_indices, trans_indices)):
		    #print ('idx=', idx)
		    #print ('wi=', wi)
		    #print ('ti=', ti)
                    # Form the new hypothesis by appending new word to the left hyp
                    new_hyp_lem = hyp_samples_lem[ti] + [wi[0]]
                    new_hyp_fact = hyp_samples_fact[ti] + [wi[1]]
                    new_ali = hyp_alignments[ti] + [alphas[ti]]
		    if wi[0] == 0:
			# <eos> found in lemmas, separate out finished hypotheses
			final_sample_lem.append(new_hyp_lem)
			final_score_lem.append(costs_lem[idx])
			final_sample_fact.append(new_hyp_fact)
			final_score_fact.append(costs_fact[idx])
			#print (idx,'final_sample_lem=',idx_to_sent(self.trg_idict, new_hyp_samples_lem[idx]), len(new_hyp_samples_lem[idx]))
			#print (idx,'final_sample_fact=',idx_to_sent(self.trgmult_idict, new_hyp_samples_fact[idx]), len(new_hyp_samples_fact[idx]))
			final_alignments.append(new_ali)
		    else:
			# Add formed hypothesis to the new hypotheses list
			new_hyp_scores_lem.append(costs_lem[idx])
			new_hyp_scores_fact.append(costs_fact[idx])
			# We get the same state from lemmas and factors
			hyp_states.append(next_state[ti])
			#print ('new_hyp_states=', new_hyp_states)
			# first position is the lemma and the second the factors
			new_hyp_samples_lem.append(new_hyp_lem)
			#print (idx,'new_hyp_sample_lem=',idx_to_sent(self.trg_idict, new_hyp_samples_lem[idx]), len(new_hyp_samples_lem[idx]))
			new_hyp_samples_fact.append(new_hyp_fact)
			#print (idx,'new_hyp_samples_fact=',idx_to_sent(self.trgmult_idict, new_hyp_samples_fact[idx]), len(new_hyp_samples_fact[idx]))
			new_hyp_alignments.append(new_ali)
			live_beam += 1


		hyp_scores_lem  = np.array(new_hyp_scores_lem, dtype=FLOAT)
		hyp_scores_fact = np.array(new_hyp_scores_fact, dtype=FLOAT)
		hyp_samples_lem = new_hyp_samples_lem
		hyp_samples_fact = new_hyp_samples_fact
		#print ('hyp_samples_lem=', hyp_samples_lem)
		#print ('hyp_samples_fact=', hyp_samples_fact)
		#print ('hyp_scores_fact=', hyp_scores_fact)
		#print ('final_sample_lem=', final_sample_lem)
		#print ('final_score=', final_score)
		#print ('final_sample_fact=', final_sample_fact)
		#print ('final_score_fact=', final_score_fact)
		hyp_alignments = new_hyp_alignments

		if live_beam == 0:
		    break

		# Take the idxs of each hyp's last word
		next_w_lem = np.array([w[-1] for w in hyp_samples_lem])
		next_w_fact = np.array([w[-1] for w in hyp_samples_fact])
		next_state = np.array(hyp_states, dtype=FLOAT)
		tiled_ctx   = np.tile(ctx0, [live_beam, 1])

	    # dump every remaining hypotheses
	    #if live_beam > 0:
	    for idx in xrange(live_beam):
	        final_score_lem.append(hyp_scores_lem[idx])
	        final_sample_lem.append(hyp_samples_lem[idx])
	        final_sample_fact.append(hyp_samples_fact[idx])
	        final_score_fact.append(hyp_scores_fact[idx])
	        final_alignments.append(hyp_alignments[idx])

	    final_score = []
	    #for b in range(beam_size):
	    for b in range(len(final_score_lem)):
		final_score.append(final_score_lem[b] + final_score_fact[b])
		#print ('score=', final_score_lem[b] + final_score_fact[b])
		#print (b,'final_sample_lem=',idx_to_sent(self.trg_idict, final_sample_lem[b]), 'len=',len(final_sample_lem[b]))
		#print (b,'final_sample_fact=',idx_to_sent(self.trgmult_idict, final_sample_lem[b]), 'len=', len(final_sample_lem[b]))
	    #print ('final_score=', final_score)
	    #print ('final_sample_lem=', final_sample_lem)
	    #print ('final_sample_fact=', final_sample_fact)
	    #print ('final_score_lem=', final_score_lem)
	    #print ('final_score_fact=', final_score_fact)
	    #print ('-------------------------------------------------')
	    return final_sample_lem, final_score, final_alignments, final_sample_fact

    '''

    def beam_search(self, inputs, beam_size=12, maxlen=50, suppress_unks=False, **kwargs):

	    # Final results and their scores
	    final_sample_lem = []
	    final_score_lem = []
	    final_sample_fact = []
	    final_score_fact = []
	    final_alignments = []

	    # Initially we have one empty hypothesis with a score of 0
	    hyp_scores  = np.zeros(1).astype(FLOAT)
	    hyp_scores_lem  = np.zeros(1).astype(FLOAT)
	    hyp_scores_fact  = np.zeros(1).astype(FLOAT)
	    hyp_samples_lem = [[]]
	    hyp_samples_fact = [[]]
	    hyp_alignments  = [[]]

	    # get initial state of decoder rnn and encoder context vectors
	    # ctx0: the set of context vectors leading to the next_state
	    # with a shape of (n_words x 1 x ctx_dim)
	    # next_state: mean context vector (ctx0.mean()) passed through FF with a final
	    # shape of (1 x 1 x ctx_dim)
	    # The 2 outputs have the same next_state
	    next_state, ctx0 = self.f_init(inputs[0])

	    # Beginning-of-sentence indicator is -1
	    next_w_lem = -1 * np.ones((1,)).astype(INT)
	    next_w_fact = -1 * np.ones((1,)).astype(INT)

	    # maxlen or 3 times source length
	    #maxlen = min(maxlen, inputs[0].shape[0] * 3)
	    maxlen = inputs[0].shape[0] * 3

            # Always starts with the initial tstep's context vectors
            # e.g. we have a ctx0 of shape (n_words x 1 x ctx_dim)
            # Tiling it live_beam times makes it (n_words x live_beam x ctx_dim)
            # thus we create sth like a batch of live_beam size with every word duplicated
            # for further state expansion.
            tiled_ctx = np.tile(ctx0, [1, 1])
            live_beam = beam_size

	    for ii in xrange(maxlen):
		# Always starts with the initial tstep's context vectors
		# e.g. we have a ctx0 of shape (n_words x 1 x ctx_dim)
		# Tiling it live_beam times makes it (n_words x live_beam x ctx_dim)
		# thus we create sth like a batch of live_beam size with every word duplicated
		# for further state expansion.

		# Get next states
		# In the first iteration, we provide -1 and obtain the log_p's for the
		# first word. In the following iterations tiled_ctx becomes a batch
		# of duplicated left hypotheses. tiled_ctx is always the same except
		# the 2nd dimension as the context vectors of the source sequence
		# is always the same regardless of the decoding step.
		next_log_p_lem, _, next_log_p_fact, _, next_state, alphas = self.f_next(*[next_w_lem, next_w_fact, tiled_ctx, next_state])
                # For each f_next, we obtain a new set of alpha's for the next_w
                # for each hypothesis in the beam search

                if suppress_unks:
                    next_log_p_lem[:, 1] = -np.inf


		# Compute sum of log_p's for the current n-gram hypotheses and flatten them
		cand_scores_lem = hyp_scores_lem[:, None] - next_log_p_lem
		cand_scores_fact = hyp_scores_fact[:, None] - next_log_p_fact
		
		# Beam search improvement for factors
		# Do combination for each new hyp
		cand_costs = []
		cand_costs_lem = []
		cand_costs_fact = []
		cand_w_idx = []
		cand_trans_idx = []
		for idx, [cand_h_scores_lem, cand_h_scores_fact] in enumerate(zip(cand_scores_lem, cand_scores_fact)):
		    # Take the best beam_size-dead_beam hypotheses
		    ranks_lem = cand_h_scores_lem.argpartition(live_beam-1)[:live_beam]
		    # if beam size if bigger than factors vocab, use vocab size as beam size for it
		    if live_beam > self.n_words_trgmult:
			ranks_fact = cand_h_scores_fact.argpartition(self.n_words_trgmult-1)[:self.n_words_trgmult]
		    else:
			ranks_fact = cand_h_scores_fact.argpartition(live_beam-1)[:live_beam]
		    # Get their costs
		    costs_h_lem = cand_h_scores_lem[ranks_lem]
		    costs_h_fact = cand_h_scores_fact[ranks_fact]
		    word_indices_lem = ranks_lem % self.n_words_trg
		    word_indices_fact = ranks_fact % self.n_words_trgmult
		    
		    # Sum the logp's of lemmas and factors and keep the best ones
		    cand_h_costs = []
		    cand_h_costs_lem = []
		    cand_h_costs_fact = []
		    cand_h_w_idx = []
		    for l in range(live_beam):
			if live_beam > self.n_words_trgmult:
			    for f in range(self.n_words_trgmult):
				cand_h_costs.append(costs_h_lem[l]+ costs_h_fact[f])
				cand_h_costs_lem.append(costs_h_lem[l])
				cand_h_costs_fact.append(costs_h_fact[f])
				# keep the word indexes of both outputs
				cand_h_w_idx.append([word_indices_lem[l], word_indices_fact[f]])
			else:
			    for f in range(live_beam):
				cand_h_costs.append(costs_h_lem[l]+ costs_h_fact[f])
				cand_h_costs_lem.append(costs_h_lem[l])
				cand_h_costs_fact.append(costs_h_fact[f])
				# keep the word indexes of both outputs
				cand_h_w_idx.append([word_indices_lem[l], word_indices_fact[f]])
		    # We convert the merged lists to np arrays and prune with the best costs and get indices of the nbest
		    cand_h_costs = np.array(cand_h_costs)
		    cand_h_costs_lem = np.array(cand_h_costs_lem)
		    cand_h_costs_fact = np.array(cand_h_costs_fact)
		    cand_h_w_idx = np.array(cand_h_w_idx)
		    ranks_h_costs = cand_h_costs.argsort()[:(live_beam)]
		    
		    # We append the beam_size hyps
		    cand_costs.append(cand_h_costs[ranks_h_costs])
		    cand_costs_lem.append(cand_h_costs_lem[ranks_h_costs])
		    cand_costs_fact.append(cand_h_costs_fact[ranks_h_costs])
		    word_h_indices = cand_h_w_idx[ranks_h_costs]
		    # We cannot flatten later this array, we need pair elements
		    for w in word_h_indices:
			cand_w_idx.append(w)
		    trans_h_indices = []
		    trans_h_indices = beam_size * [idx]
		    trans_h_indices = np.array(trans_h_indices)
		    cand_trans_idx.append(trans_h_indices)

		# We convert the merged lists to np arrays and prune with the best costs and get indices of the nbest
		cand_costs = np.array(cand_costs)
		cand_costs_lem = np.array(cand_costs_lem)
		cand_costs_fact = np.array(cand_costs_fact)
		cand_w_idx = np.array(cand_w_idx)
		cand_trans_idx = np.array(cand_trans_idx)
		cand_flat_costs = cand_costs.flatten()
		cand_flat_costs_lem = cand_costs_lem.flatten()
		cand_flat_costs_fact = cand_costs_fact.flatten()
		ranks_costs = cand_flat_costs.argsort()[:(live_beam)]
		costs = cand_flat_costs[ranks_costs]
		costs_lem = cand_flat_costs_lem[ranks_costs]
		costs_fact = cand_flat_costs_fact[ranks_costs]
		word_indices = cand_w_idx[ranks_costs]
		cand_trans_idx_flat = cand_trans_idx.flatten()
		trans_indices = cand_trans_idx_flat[ranks_costs]


		# New states, scores and samples
		live_beam = 0
		# We have shared scores for both outputs after the last pruning
		new_hyp_scores = []
		new_hyp_samples_lem = []
		new_hyp_scores_lem = []
		new_hyp_samples_fact = []
		# Using the EOS of lemmas for factors
		new_hyp_scores_fact = []
		new_hyp_alignments  = []

		# This will be the new next states in the next iteration
	        hyp_states = []

		# Iterate over the hypotheses and add them to new_* lists
		# We have common next_state
		for idx, [wi, ti] in enumerate(zip(word_indices, trans_indices)):
                    # Form the new hypothesis by appending new word to the left hyp
                    new_hyp_lem = hyp_samples_lem[ti] + [wi[0]]
                    new_hyp_fact = hyp_samples_fact[ti] + [wi[1]]
                    new_ali = hyp_alignments[ti] + [alphas[ti]]
		    if wi[0] == 0:
			# <eos> found in lemmas, separate out finished hypotheses
			final_sample_lem.append(new_hyp_lem)
			final_score_lem.append(costs_lem[idx])
			final_sample_fact.append(new_hyp_fact)
			final_score_fact.append(costs_fact[idx])
			final_alignments.append(new_ali)
		    else:
			# Add formed hypothesis to the new hypotheses list
			new_hyp_scores_lem.append(costs_lem[idx])
			new_hyp_scores_fact.append(costs_fact[idx])
			# We get the same state from lemmas and factors
			hyp_states.append(next_state[ti])
			# first position is the lemma and the second the factors
			new_hyp_samples_lem.append(new_hyp_lem)
			new_hyp_samples_fact.append(new_hyp_fact)
			new_hyp_alignments.append(new_ali)
			live_beam += 1


		hyp_scores_lem  = np.array(new_hyp_scores_lem, dtype=FLOAT)
		hyp_scores_fact = np.array(new_hyp_scores_fact, dtype=FLOAT)
		hyp_samples_lem = new_hyp_samples_lem
		hyp_samples_fact = new_hyp_samples_fact
		hyp_alignments = new_hyp_alignments

		if live_beam == 0:
		    break

		# Take the idxs of each hyp's last word
		next_w_lem = np.array([w[-1] for w in hyp_samples_lem])
		next_w_fact = np.array([w[-1] for w in hyp_samples_fact])
		next_state = np.array(hyp_states, dtype=FLOAT)
		tiled_ctx   = np.tile(ctx0, [live_beam, 1])

	    # dump every remaining hypotheses
	    #if live_beam > 0:
	    for idx in xrange(live_beam):
	        final_score_lem.append(hyp_scores_lem[idx])
	        final_sample_lem.append(hyp_samples_lem[idx])
	        final_sample_fact.append(hyp_samples_fact[idx])
	        final_score_fact.append(hyp_scores_fact[idx])
	        final_alignments.append(hyp_alignments[idx])

	    final_score = []
	    for b in range(beam_size):
		final_score.append(final_score_lem[b] + final_score_fact[b])
	    return final_sample_lem, final_score, final_alignments, final_sample_fact


    def beam_search(self, inputs, beam_size=12, maxlen=50, suppress_unks=False, **kwargs):

	    # Final results and their scores
	    final_sample_lem = []
	    final_score_lem = []
	    final_sample_fact = []
	    final_score_fact = []
	    final_alignments = []

	    # Initially we have one empty hypothesis with a score of 0
	    hyp_scores  = np.zeros(1).astype(FLOAT)
	    hyp_scores_lem  = np.zeros(1).astype(FLOAT)
	    hyp_scores_fact  = np.zeros(1).astype(FLOAT)
	    hyp_samples_lem = [[]]
	    hyp_samples_fact = [[]]
	    hyp_alignments  = [[]]

	    # get initial state of decoder rnn and encoder context vectors
	    # ctx0: the set of context vectors leading to the next_state
	    # with a shape of (n_words x 1 x ctx_dim)
	    # next_state: mean context vector (ctx0.mean()) passed through FF with a final
	    # shape of (1 x 1 x ctx_dim)
	    # The 2 outputs have the same next_state
	    next_state, ctx0 = self.f_init(inputs[0])

	    # Beginning-of-sentence indicator is -1
	    next_w_lem = -1 * np.ones((1,)).astype(INT)
	    next_w_fact = -1 * np.ones((1,)).astype(INT)

	    # maxlen or 3 times source length
	    #maxlen = min(maxlen, inputs[0].shape[0] * 3)
	    maxlen = inputs[0].shape[0] * 3

            # Always starts with the initial tstep's context vectors
            # e.g. we have a ctx0 of shape (n_words x 1 x ctx_dim)
            # Tiling it live_beam times makes it (n_words x live_beam x ctx_dim)
            # thus we create sth like a batch of live_beam size with every word duplicated
            # for further state expansion.
            tiled_ctx = np.tile(ctx0, [1, 1])
            live_beam = beam_size

	    for ii in xrange(maxlen):
		# Always starts with the initial tstep's context vectors
		# e.g. we have a ctx0 of shape (n_words x 1 x ctx_dim)
		# Tiling it live_beam times makes it (n_words x live_beam x ctx_dim)
		# thus we create sth like a batch of live_beam size with every word duplicated
		# for further state expansion.

		# Get next states
		# In the first iteration, we provide -1 and obtain the log_p's for the
		# first word. In the following iterations tiled_ctx becomes a batch
		# of duplicated left hypotheses. tiled_ctx is always the same except
		# the 2nd dimension as the context vectors of the source sequence
		# is always the same regardless of the decoding step.
		next_log_p_lem, _, next_log_p_fact, _, next_state, alphas = self.f_next(*[next_w_lem, next_w_fact, tiled_ctx, next_state])
                # For each f_next, we obtain a new set of alpha's for the next_w
                # for each hypothesis in the beam search

                if suppress_unks:
                    next_log_p_lem[:, 1] = -np.inf


		# Compute sum of log_p's for the current n-gram hypotheses and flatten them
		cand_scores_lem = hyp_scores_lem[:, None] - next_log_p_lem
		cand_scores_fact = hyp_scores_fact[:, None] - next_log_p_fact
		
		# Beam search improvement for factors
		# Do combination for each new hyp
		cand_costs = []
		cand_costs_lem = []
		cand_costs_fact = []
		cand_w_idx = []
		cand_trans_idx = []
		for idx, [cand_h_scores_lem, cand_h_scores_fact] in enumerate(zip(cand_scores_lem, cand_scores_fact)):
		    # Take the best beam_size-dead_beam hypotheses
		    ranks_lem = cand_h_scores_lem.argpartition(live_beam-1)[:live_beam]
		    # if beam size if bigger than factors vocab, use vocab size as beam size for it
		    if live_beam > self.n_words_trgmult:
			ranks_fact = cand_h_scores_fact.argpartition(self.n_words_trgmult-1)[:self.n_words_trgmult]
		    else:
			ranks_fact = cand_h_scores_fact.argpartition(live_beam-1)[:live_beam]
		    # Get their costs
		    costs_h_lem = cand_h_scores_lem[ranks_lem]
		    costs_h_fact = cand_h_scores_fact[ranks_fact]
		    word_indices_lem = ranks_lem % self.n_words_trg
		    word_indices_fact = ranks_fact % self.n_words_trgmult
		    
		    # Sum the logp's of lemmas and factors and keep the best ones
		    cand_h_costs = []
		    cand_h_costs_lem = []
		    cand_h_costs_fact = []
		    cand_h_w_idx = []
		    for l in range(live_beam):
			if live_beam > self.n_words_trgmult:
			    for f in range(self.n_words_trgmult):
				cand_h_costs.append(costs_h_lem[l]+ costs_h_fact[f])
				cand_h_costs_lem.append(costs_h_lem[l])
				cand_h_costs_fact.append(costs_h_fact[f])
				# keep the word indexes of both outputs
				cand_h_w_idx.append([word_indices_lem[l], word_indices_fact[f]])
			else:
			    for f in range(live_beam):
				cand_h_costs.append(costs_h_lem[l]+ costs_h_fact[f])
				cand_h_costs_lem.append(costs_h_lem[l])
				cand_h_costs_fact.append(costs_h_fact[f])
				# keep the word indexes of both outputs
				cand_h_w_idx.append([word_indices_lem[l], word_indices_fact[f]])
		    # We convert the merged lists to np arrays and prune with the best costs and get indices of the nbest
		    cand_h_costs = np.array(cand_h_costs)
		    cand_h_costs_lem = np.array(cand_h_costs_lem)
		    cand_h_costs_fact = np.array(cand_h_costs_fact)
		    cand_h_w_idx = np.array(cand_h_w_idx)
		    ranks_h_costs = cand_h_costs.argsort()[:(live_beam)]
		    
		    # We append the beam_size hyps
		    cand_costs.append(cand_h_costs[ranks_h_costs])
		    cand_costs_lem.append(cand_h_costs_lem[ranks_h_costs])
		    cand_costs_fact.append(cand_h_costs_fact[ranks_h_costs])
		    word_h_indices = cand_h_w_idx[ranks_h_costs]
		    # We cannot flatten later this array, we need pair elements
		    for w in word_h_indices:
			cand_w_idx.append(w)
		    trans_h_indices = []
		    trans_h_indices = beam_size * [idx]
		    trans_h_indices = np.array(trans_h_indices)
		    cand_trans_idx.append(trans_h_indices)

		# We convert the merged lists to np arrays and prune with the best costs and get indices of the nbest
		cand_costs = np.array(cand_costs)
		cand_costs_lem = np.array(cand_costs_lem)
		cand_costs_fact = np.array(cand_costs_fact)
		cand_w_idx = np.array(cand_w_idx)
		cand_trans_idx = np.array(cand_trans_idx)
		cand_flat_costs = cand_costs.flatten()
		cand_flat_costs_lem = cand_costs_lem.flatten()
		cand_flat_costs_fact = cand_costs_fact.flatten()
		ranks_costs = cand_flat_costs.argsort()[:(live_beam)]
		costs = cand_flat_costs[ranks_costs]
		costs_lem = cand_flat_costs_lem[ranks_costs]
		costs_fact = cand_flat_costs_fact[ranks_costs]
		word_indices = cand_w_idx[ranks_costs]
		cand_trans_idx_flat = cand_trans_idx.flatten()
		trans_indices = cand_trans_idx_flat[ranks_costs]


		# New states, scores and samples
		live_beam = 0
		# We have shared scores for both outputs after the last pruning
		new_hyp_scores = []
		new_hyp_samples_lem = []
		new_hyp_scores_lem = []
		new_hyp_samples_fact = []
		# Using the EOS of lemmas for factors
		new_hyp_scores_fact = []
		new_hyp_alignments  = []

		# This will be the new next states in the next iteration
	        hyp_states = []

		# Iterate over the hypotheses and add them to new_* lists
		# We have common next_state
		for idx, [wi, ti] in enumerate(zip(word_indices, trans_indices)):
                    # Form the new hypothesis by appending new word to the left hyp
                    new_hyp_lem = hyp_samples_lem[ti] + [wi[0]]
                    new_hyp_fact = hyp_samples_fact[ti] + [wi[1]]
                    new_ali = hyp_alignments[ti] + [alphas[ti]]
		    if wi[0] == 0:
			# <eos> found in lemmas, separate out finished hypotheses
			final_sample_lem.append(new_hyp_lem)
			final_score_lem.append(costs_lem[idx])
			final_sample_fact.append(new_hyp_fact)
			final_score_fact.append(costs_fact[idx])
			final_alignments.append(new_ali)
		    else:
			# Add formed hypothesis to the new hypotheses list
			new_hyp_scores_lem.append(costs_lem[idx])
			new_hyp_scores_fact.append(costs_fact[idx])
			# We get the same state from lemmas and factors
			hyp_states.append(next_state[ti])
			# first position is the lemma and the second the factors
			new_hyp_samples_lem.append(new_hyp_lem)
			new_hyp_samples_fact.append(new_hyp_fact)
			new_hyp_alignments.append(new_ali)
			live_beam += 1


		hyp_scores_lem  = np.array(new_hyp_scores_lem, dtype=FLOAT)
		hyp_scores_fact = np.array(new_hyp_scores_fact, dtype=FLOAT)
		hyp_samples_lem = new_hyp_samples_lem
		hyp_samples_fact = new_hyp_samples_fact
		hyp_alignments = new_hyp_alignments

		if live_beam == 0:
		    break

		# Take the idxs of each hyp's last word
		next_w_lem = np.array([w[-1] for w in hyp_samples_lem])
		next_w_fact = np.array([w[-1] for w in hyp_samples_fact])
		next_state = np.array(hyp_states, dtype=FLOAT)
		tiled_ctx   = np.tile(ctx0, [live_beam, 1])

	    # dump every remaining hypotheses
	    #if live_beam > 0:
	    for idx in xrange(live_beam):
	        final_score_lem.append(hyp_scores_lem[idx])
	        final_sample_lem.append(hyp_samples_lem[idx])
	        final_sample_fact.append(hyp_samples_fact[idx])
	        final_score_fact.append(hyp_scores_fact[idx])
	        final_alignments.append(hyp_alignments[idx])

	    final_score = []
	    for b in range(beam_size):
		final_score.append(final_score_lem[b] + final_score_fact[b])
	    return final_sample_lem, final_score, final_alignments, final_sample_fact

