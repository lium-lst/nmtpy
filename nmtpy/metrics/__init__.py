# -*- coding: utf-8 -*-
import os
import operator
import numpy as np

from .bleu              import MultiBleuScorer
from .meteor            import METEORScorer
from .factors2wordbleu  import Factors2word
from .mtevalbleu        import MTEvalV13aBLEUScorer
from .external          import ExternalScorer

comparators = {
        'bleu'      : (max, operator.gt, 0),
        'bleu_v13a' : (max, operator.gt, 0),
        'meteor'    : (max, operator.gt, 0),
        'cider'     : (max, operator.gt, 0),
        'rouge'     : (max, operator.gt, 0),
        'loss'      : (min, operator.lt, -1),
        'ter'       : (min, operator.lt, -1),
    }

def get_scorer(scorer):
    scorers = {
                'meteor'      : METEORScorer,
                'bleu'        : MultiBleuScorer,
                'bleu_v13a'   : MTEvalV13aBLEUScorer,
                'factors2word': Factors2word,
              }

    if scorer in scorers:
        # A defined metric
        return scorers[scorer]()
    elif scorer.startswith(('/', '~')):
        # External script
        return ExternalScorer(os.path.expanduser(scorer))

def is_last_best(name, history, min_delta):
    """Checks whether the last element is the best score so far
    by taking into account an absolute improvement threshold min_delta."""
    if len(history) == 1:
        # If first validation, return True to save it
        return True

    new_value = history[-1]

    # bigger is better
    if name.startswith(('bleu', 'meteor', 'cider', 'rouge')):
        cur_best = max(history[:-1])
        return new_value > cur_best and abs(new_value - cur_best) >= (min_delta - 1e-5)
    # lower is better
    elif name in ['loss', 'px', 'ter']:
        cur_best = min(history[:-1])
        return new_value < cur_best and abs(new_value - cur_best) >= (min_delta - 1e-5)

def find_best(name, history):
    """Returns the best idx and value for the given metric."""
    history = np.array(history)
    if name.startswith(('bleu', 'meteor', 'cider', 'rouge')):
        best_idx = np.argmax(history)
    elif name in ['loss', 'px', 'ter']:
        best_idx = np.argmin(history)

    # Validation periods start from 1
    best_val = history[best_idx]
    return (best_idx + 1), best_val
