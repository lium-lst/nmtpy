import operator

from .bleu   import MultiBleuScorer
from .meteor import METEORScorer
from .factors2wordbleu import Factors2word

import numpy as np

comparators = {
        'bleu'   : (max, operator.gt, 0),
        'meteor' : (max, operator.gt, 0),
        'cider'  : (max, operator.gt, 0),
        'rouge'  : (max, operator.gt, 0),
        'loss'   : (min, operator.lt, -1),
        'ter'    : (min, operator.lt, -1),
    }

def get_scorer(scorer):
    scorers = {
                'meteor'      : METEORScorer,
                'bleu'        : MultiBleuScorer,
                'factors2word': Factors2word,
              }

    return scorers[scorer]

def is_last_best(name, history):
    """Checks whether the last element is the best score so far."""
    if len(history) == 1:
        # If first validation, return True to save it
        return True

    new_value = history[-1]
    if name in ['bleu', 'meteor', 'cider', 'rouge']:
        # bigger is better
        return new_value > max(history[:-1])
    elif name in ['loss', 'px', 'ter']:
        # lower is better
        return new_value < min(history[:-1])

def find_best(name, history):
    """Returns the best idx and value for the given metric."""
    history = np.array(history)
    if name in ['bleu', 'meteor', 'cider', 'rouge']:
        best_idx = np.argmax(history)
    elif name in ['loss', 'px', 'ter']:
        best_idx = np.argmin(history)

    # Validation periods start from 1
    best_val = history[best_idx]
    return (best_idx + 1), best_val
