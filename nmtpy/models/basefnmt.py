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
from .attention import Model
from ..sysutils import readable_size, get_temp_file, get_valid_evaluation

class Model(BaseModel):
    def __init__(self, seed, logger, **kwargs):
        # Call parent's init first
        super(Model, self).__init__(**kwargs)

        # Set logger
        self._logger = logger

        # Use GRU by default as encoder
        self.enc_type = kwargs.get('enc_type', 'gru')

        # Do we apply layer normalization to GRU?
        self.layer_norm = kwargs.get('layer_norm', False)

        # Shuffle mode (default: No shuffle)
        self.shuffle_mode = kwargs.get('shuffle_mode', 'simple')

        # How to initialize CGRU
        self.init_cgru = kwargs.get('init_cgru', 'text')

        # Get dropout parameters
        # Let's keep the defaults as 0 to not use dropout
        # You can adjust those from your conf files.
        self.emb_dropout = kwargs.get('emb_dropout', 0.)
        self.ctx_dropout = kwargs.get('ctx_dropout', 0.)
        self.out_dropout = kwargs.get('out_dropout', 0.)

        # Number of additional GRU encoders for source sentences
        self.n_enc_layers  = kwargs.get('n_enc_layers' , 1)

        # Use a single embedding matrix for target words?
        self.tied_trg_emb = kwargs.get('tied_trg_emb', False)
        
        self.factors = kwargs.get('factors', None)

        # Load dictionaries
        if 'src_dict' in kwargs:
            # Already passed through kwargs (nmt-translate)
            self.src_dict = kwargs['src_dict']
            # Invert dict
            src_idict = invert_dictionary(self.src_dict)
        else:
            # Load them from pkl files
            self.src_dict, src_idict = load_dictionary(kwargs['dicts']['src'])

        if 'trg1_dict' in kwargs:
            # Already passed through kwargs (nmt-translate)
            self.trg_dict = kwargs['trglem_dict']
            # Invert dict
            trg_idict = invert_dictionary(self.trg_dict)
        else:
            # Load them from pkl files
            self.trg_dict, trg_idict = load_dictionary(kwargs['dicts']['trg1'])
        if 'trg2_dict' in kwargs:
            # Already passed through kwargs (nmt-translate)
            self.trgfact_dict = kwargs['trgfact_dict']
            # Invert dict
            trgfact_idict = invert_dictionary(self.trgfact_dict)
        else:
            # Load them from pkl files
            self.trgfact_dict, trgfact_idict = load_dictionary(kwargs['dicts']['trg2'])

        # Load constraints on factor predictions (Franck).
        # The loaded file contains on each line elements
        # separated by space. The 1st element is the lemma,
        # all others are allowed factors for the lemma. Ex.:
        # dog noun+singular noun+plural
#        global fact_constraints
#        fact_constraints = defaultdict(lambda: np.array(range(len(trgfact_idict))))
#        try:
            # Set the path to file with factor constraints
            # dictionary with lemma and factors, each line lemma factor1 factor2 factor3
#            const_file = open('/users/limsi_nmt/burlot/prog/wmt17/constraints.en2cx.bpe')
            #const_file = open('/lium/buster1/garcia/workspace/scripts/latvian/constraints.lv')
            #const_file = open('/lium/buster1/garcia/workspace/scripts/czech/constraints.cs')
            #const_file = open('/lium/buster1/garcia/workspace/scripts/czech/constraints.bpe.cs')
            #const_file = open('/lium/buster1/garcia/workspace/scripts/czech/constraints.noprev.bpe.cs')
#            print("Constrained search", const_file)
#        except FileNotFoundError:
#            print("File with factor constraints not found: unconstrained search")
#            const_file = []
#        for line in const_file:
#            line = line.split()
#            try:
#                lem = self.trg_dict[line[0]]
#            except KeyError:
#                continue
#            facts = [self.trgfact_dict.get(f, self.trgfact_dict['<unk>']) for f in line[1:]]
#            fact_constraints[lem] = np.array(facts)

        # Limit shortlist sizes
        self.n_words_src = min(self.n_words_src, len(self.src_dict)) \
                if self.n_words_src > 0 else len(self.src_dict)
        self.n_words_trg1 = min(self.n_words_trg1, len(self.trg_dict)) \
                if self.n_words_trg1 > 0 else len(self.trg_dict)
        self.n_words_trg2 = min(self.n_words_trg2, len(self.trgfact_dict)) \
                if self.n_words_trg2 > 0 else len(self.trgfact_dict)

        # Create options. This will saved as .pkl
        self.set_options(self.__dict__)

        self.src_idict = src_idict
        self.trg_idict = trg_idict
        self.trgfact_idict = trgfact_idict

        # Context dimensionality is 2 times RNN since we use Bi-RNN
        self.ctx_dim = 2 * self.rnn_dim

        # Set the seed of Theano RNG
        self.set_trng(seed)

        # We call this once to setup dropout mechanism correctly
        self.set_dropout(False)

    def run_beam_search(self, beam_size=12, n_jobs=8, metric='bleu', mode='beamsearch', valid_mode='single', f_valid_out=None):
        """Save model under /tmp for passing it to nmt-translate-factors."""
        # Save model temporarily
        with get_temp_file(suffix=".npz", delete=True) as tmpf:
            self.save(tmpf.name)

            result = get_valid_evaluation(tmpf.name,
                                          trans_cmd='nmt-translate-factors',
                                          beam_size=beam_size,
                                          n_jobs=n_jobs,
                                          metric=metric,
                                          f_valid_out=f_valid_out,
                                          factors=self.factors)
        lem_bleu_str, lem_bleu = result['out1']
        self._logger.info("Out1: %s" % lem_bleu_str)
        fact_bleu_str, fact_bleu = result['out2']
        self._logger.info("Out2: %s" % fact_bleu_str)

        return {metric: result[metric]}

    @staticmethod
    def beam_search(inputs, f_inits, f_nexts, beam_size=12, maxlen=50, suppress_unks=False, fact_constraints=[], **kwargs):
        #global fact_constraints
        # Final results and their scores
        final_sample_lem = []
        final_score_lem = []
        final_sample_fact = []
        final_score_fact = []
        final_alignments = []

        # Initially we have one empty hypothesis with a score of 0
        hyp_alignments  = [[]]
        hyp_samples_lem = [[]]
        hyp_samples_fact = [[]]
        hyp_scores  = np.zeros(1).astype(FLOAT)
        hyp_scores_lem  = np.zeros(1).astype(FLOAT)
        hyp_scores_fact  = np.zeros(1).astype(FLOAT)
        
        # Number of models
        n_models        = len(f_inits)
        
        # Ensembling-aware lists
        next_states     = [None] * n_models
        text_ctxs       = [None] * n_models
        aux_ctxs        = [[]] * n_models
        tiled_ctxs      = [None] * n_models
        next_log_ps_lem = [None] * n_models
        next_log_ps_fact = [None] * n_models
        alphas          = [None] * n_models
        
        for i, f_init in enumerate(f_inits):
            # Get next_state and initial contexts and save them
            # text_ctx: the set of textual annotations
            # aux_ctx: the set of auxiliary (ex: image) annotations
            # NOTE: with factors we do not use yet the images
            result = list(f_init(*inputs))
            next_states[i], text_ctxs[i], aux_ctxs[i] = result[0], result[1], result[2:]
            tiled_ctxs[i] = np.tile(text_ctxs[i], [1, 1])

        # Beginning-of-sentence indicator is -1
        next_w_lem = -1 * np.ones((1,)).astype(INT)
        next_w_fact = -1 * np.ones((1,)).astype(INT)

        # FIXME: This will break if [0] is not the src sentence, e.g. im2txt models
        maxlen = max(maxlen, inputs[0].shape[0] * 3)

        # Initial beam size
        live_beam = beam_size

        for t in range(maxlen):
            # Get next states
            # In the first iteration, we provide -1 and obtain the log_p's for the
            # first word. In the following iterations tiled_ctx becomes a batch
            # of duplicated left hypotheses. tiled_ctx is always the same except
            # the size of the 2nd dimension as the context vectors of the source
            # sequence is always the same regardless of the decoding process.
            # next_state's shape is (live_beam, rnn_dim)

            # NOTE: next_state and titled_ctx were switch before ensamble
#            next_log_p_lem, next_log_p_fact, next_state, alphas = f_nexts[0](*[next_w_lem, next_w_fact, next_state, tiled_ctx])
            #next_log_p_lem, next_log_p_fact, next_state, alphas = f_nexts[0](*[next_w_lem, next_w_fact, tiled_ctx, next_state])
            # We do this for each model
            for m, f_next in enumerate(f_nexts):
                next_log_ps_lem[m], next_log_ps_fact[m], next_states[m], alphas[m] = f_next(*([next_w_lem, next_w_fact, next_states[m], tiled_ctxs[m]] + aux_ctxs[m]))
               
                # NOTE: supress_unks does not work yet for factors
                if suppress_unks:
                    next_log_ps_lem[m][:, 1] = -np.inf

            # Compute sum of log_p's for the current n-gram hypotheses 
            cand_scores_lem = hyp_scores_lem[:, None] - sum(next_log_ps_lem)
            cand_scores_fact = hyp_scores_fact[:, None] - sum(next_log_ps_fact)
            
            # Mean alphas for the mean model (n_models > 1)
            mean_alphas = sum(alphas) / n_models
                
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
                # Get their costs
                costs_h_lem = cand_h_scores_lem[ranks_lem]
                # All models should have the same shape for ensamble so we use just the first one 
                word_indices_lem = ranks_lem % next_log_ps_lem[0].shape[1]

                # get factor constraints for each lemma selected for the beam (Franck)
                costs_h_fact = {}
                word_indices_fact = {}
                for l in word_indices_lem:
                    cost_constr_fact = cand_h_scores_fact[fact_constraints[l]]
                    # NOTE: the beam size could be higher than the fact dict
                    if live_beam < cost_constr_fact.shape[0]:
                        ranks_fact = cost_constr_fact.argpartition(live_beam-1)[:live_beam]
                    else:
                        ranks_fact = np.array(range(len(fact_constraints[l])))
                    costs_h_fact[l] = cost_constr_fact[ranks_fact]
                    word_indices_fact[l] = np.array([fact_constraints[l][n] for n in ranks_fact])
                        
                # Sum the logp's of lemmas and factors and keep the best ones
                cand_h_costs = []
                cand_h_costs_lem = []
                cand_h_costs_fact = []
                cand_h_w_idx = []
                for l in range(live_beam):
                    # cost_h_fact has different values for each lemma (Franck)
                    l_idx = word_indices_lem[l]
                    for f in range(costs_h_fact[l_idx].shape[0]):
                        cand_h_costs.append(costs_h_lem[l]+ costs_h_fact[l_idx][f])
                        cand_h_costs_lem.append(costs_h_lem[l])
                        cand_h_costs_fact.append(costs_h_fact[l_idx][f])
                        # keep the word indexes of both outputs
                        cand_h_w_idx.append([l_idx, word_indices_fact[l_idx][f]])

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
                trans_h_indices = live_beam * [idx]
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
                new_ali = hyp_alignments[ti] + [mean_alphas[ti]]
                if wi[0] == 0:
                    # <eos> found in lemmas, separate out finished hypotheses
                    final_sample_lem.append(new_hyp_lem)
                    final_score_lem.append(costs_lem[idx])
                    final_sample_fact.append(hyp_samples_fact[ti])
                    final_score_fact.append(costs_fact[idx])
                    final_alignments.append(new_ali)
                else:
                    # Add formed hypothesis to the new hypotheses list
                    new_hyp_scores_lem.append(costs_lem[idx])
                    new_hyp_scores_fact.append(costs_fact[idx])
                    # We get the same state from lemmas and factors
                    # Hidden state of the decoder for this hypothesis
                    hyp_states.append([next_state[ti] for next_state in next_states])
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
            next_states = [np.array(st, dtype=FLOAT) for st in zip(*hyp_states)]
            tiled_ctxs  = [np.tile(ctx, [live_beam, 1]) for ctx in text_ctxs]

        # dump every remaining hypotheses
        #if live_beam > 0:
        for idx in range(live_beam):
            final_score_lem.append(hyp_scores_lem[idx])
            final_sample_lem.append(hyp_samples_lem[idx])
            final_sample_fact.append(hyp_samples_fact[idx])
            final_score_fact.append(hyp_scores_fact[idx])
            final_alignments.append(hyp_alignments[idx])

        final_score = []
        for b in range(beam_size):
            final_score.append(final_score_lem[b] + final_score_fact[b])
        
        if not kwargs.get('get_att_alphas', False):
            # Don't send back alignments for nothing
            final_alignments = None
        return final_sample_lem, final_score, final_alignments, final_sample_fact

    def info(self):
        self._logger.info('Source vocabulary size: %d', self.n_words_src)
        self._logger.info('Target vocabulary size: %d', self.n_words_trg1)
        self._logger.info('Target factors vocabulary size: %d', self.n_words_trg2)
        self._logger.info('%d training samples' % self.train_iterator.n_samples)
        self._logger.info('%d validation samples' % self.valid_iterator.n_samples)
        self._logger.info('dropout (emb,ctx,out): %.2f, %.2f, %.2f' % (self.emb_dropout, self.ctx_dropout, self.out_dropout))

    def load_valid_data(self, from_translate=False):
        self.valid_ref_files = self.data['valid_trg']
        if isinstance(self.valid_ref_files, str):
            self.valid_ref_files = list([self.valid_ref_files])

        if from_translate:
            self.valid_iterator = TextIterator(
                                    mask=False,
                                    batch_size=1,
                                    file=self.data['valid_src'], dict=self.src_dict,
                                    n_words=self.n_words_src)
        else:
            # Take the first validation item for NLL computation
            self.valid_iterator = FactorsIterator(
                                    batch_size=self.batch_size,
                                    srcfile=self.data['valid_src'], srcdict=self.src_dict,
                                    trglemfile=self.data['valid_trg1'], trglemdict=self.trg_dict,
                                    trgfactfile=self.data['valid_trg2'], trgfactdict=self.trgfact_dict,
                                    #trgfile=self.valid_ref_files[0], trgdict=self.trg_dict,
                                    n_words_src=self.n_words_src, n_words_trg=self.n_words_trg1,
                                    n_words_trglem=self.n_words_trg1, n_words_trgfact=self.n_words_trg2)

        self.valid_iterator.read()

    def load_data(self):
        self.train_iterator = FactorsIterator(
                                batch_size=self.batch_size,
                                shuffle_mode=self.shuffle_mode,
                                logger=self._logger,
                                srcfile=self.data['train_src'], srcdict=self.src_dict,
                                trglemfile=self.data['train_trg1'], trglemdict=self.trg_dict,
                                trgfactfile=self.data['train_trg2'], trgfactdict=self.trgfact_dict,
                                n_words_src=self.n_words_src,
                                n_words_trglem=self.n_words_trg1, n_words_trgfact=self.n_words_trg2)

        # Prepare batches
        self.train_iterator.read()
        self.load_valid_data()

    ###################################################################
    # The following methods can be redefined in child models inheriting
    # from this basic Attention model.
    ###################################################################
    # TODO we could include common things in the following functions
    def init_params(self):

        pass

    def build(self):

        pass

    def build_sampler(self, **kwargs):

        pass
