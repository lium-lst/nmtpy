# -*- coding: utf-8 -*-
import re

class Filter(object):
    """Common Filter class for post-processing sentences."""
    def __call__(self, inp):
        if isinstance(inp, str):
            # Apply to single sentence
            return self.process(inp)
        else:
            # Assume a sequence and apply to each
            return [self.process(e) for e in inp]

    def process(self, s):
        # Derived classes should implement this method
        return s

class CompoundFilter(Filter):
    """Filters out fillers from compound splitted sentences."""
    def process(self, s):
        return s.replace(" @@ ", "").replace(" @@", "").replace(" @", "").replace("@ ", "")

class BPEFilter(Filter):
    """Filters out fillers from BPE applied sentences."""
    def process(self, s):
        # The first replace misses lines ending with @@
        # like 'foo@@ bar Hotel@@'
        return s.replace("@@ ", "").replace("@@", "")

class DesegmentFilter(Filter):
    """Converts Turkish segmentations of <tag:morpheme> to normal form."""
    def process(self, s):
        return re.sub(' *<.*?:(.*?)>', '\\1', s)

class Char2Words(Filter):
    """Converts a space delimited character sequence to
    normal word form. The output will be non-tokenized."""
    def process(self, s):
        return s.replace(' ', '').replace('<s>', ' ').strip()

def get_filter(name):
    filters = {
                "bpe"          : BPEFilter(),
                "char2words"   : Char2Words(),
                "compound"     : CompoundFilter(),
                "desegment"    : DesegmentFilter(),
              }
    return filters.get(name, None)
