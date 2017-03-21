from .bleu   import MultiBleuScorer
from .meteor import METEORScorer
from .factors2wordbleu import Factors2word

def get_scorer(scorer):
    scorers = {
                'meteor': METEORScorer,
                'bleu'  : MultiBleuScorer,
		'factors2word': Factors2word,
              }

    if scorer == 'all':
        return scorers
    else:
        return scorers[scorer]
