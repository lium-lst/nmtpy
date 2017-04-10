# -*- coding: utf-8 -*-
import subprocess
import pkg_resources

from .metric import Metric

BLEU_SCRIPT = pkg_resources.resource_filename('nmtpy', 'external/multi-bleu.perl')

class BLEUScore(Metric):
    def __init__(self, score=None):
        super(BLEUScore, self).__init__(score)
        self.name = "BLEU"
        if score:
            self.score = float(score.split()[2][:-1])
            self.score_str = score.replace('BLEU = ', '')

"""MultiBleuScorer class."""
class MultiBleuScorer(object):
    def __init__(self, lowercase=False):
        # For multi-bleu.perl we give the reference(s) files as argv,
        # while the candidate translations are read from stdin.
        self.lowercase = lowercase
        self.__cmdline = [BLEU_SCRIPT]
        if self.lowercase:
            self.__cmdline.append("-lc")

    def compute(self, refs, hypfile):
        cmdline = self.__cmdline[:]

        # Make reference files a list
        refs = [refs] if isinstance(refs, str) else refs
        cmdline.extend(refs)

        hypstring = None
        with open(hypfile, "r") as fhyp:
            hypstring = fhyp.read().rstrip()

        score = subprocess.run(cmdline, stdout=subprocess.PIPE,
                               input=hypstring, universal_newlines=True).stdout.splitlines()
        if len(score) == 0:
            return BLEUScore()
        else:
            return BLEUScore(score[0].rstrip("\n"))
