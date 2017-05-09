# -*- coding: utf-8 -*-
import numpy as np

from ..sysutils   import fopen
from .iterator    import Iterator
from .homogeneous import HomogeneousData

"""Parallel text iterator for translation data."""
class BiTextIterator(Iterator):
    def __init__(self, batch_size, seed=1234, mask=True, shuffle_mode=None, logger=None, **kwargs):
        super(BiTextIterator, self).__init__(batch_size, seed, mask, shuffle_mode, logger)

        assert 'srcfile' in kwargs, "Missing argument srcfile"
        assert 'trgfile' in kwargs, "Missing argument trgfile"
        assert 'srcdict' in kwargs, "Missing argument srcdict"
        assert 'trgdict' in kwargs, "Missing argument trgdict"
        assert batch_size > 1, "Batch size should be > 1"

        self._print('Shuffle mode: %s' % shuffle_mode)

        self.srcfile = kwargs['srcfile']
        self.trgfile = kwargs['trgfile']
        self.srcdict = kwargs['srcdict']
        self.trgdict = kwargs['trgdict']

        self.n_words_src = kwargs.get('n_words_src', 0)
        self.n_words_trg = kwargs.get('n_words_trg', 0)

        self.src_name = kwargs.get('src_name', 'x')
        self.trg_name = kwargs.get('trg_name', 'y')

        self._keys = [self.src_name]
        if self.mask:
            self._keys.append("%s_mask" % self.src_name)

        self._keys.append(self.trg_name)
        if self.mask:
            self._keys.append("%s_mask" % self.trg_name)

    def read(self):
        seqs = []
        sf = fopen(self.srcfile, 'r')
        tf = fopen(self.trgfile, 'r')

        for idx, (sline, tline) in enumerate(zip(sf, tf)):
            sline = sline.strip()
            tline = tline.strip()

            # Exception if empty line found
            if sline == "" or tline == "":
                continue

            sseq = [self.srcdict.get(w, 1) for w in sline.split(' ')]
            tseq = [self.trgdict.get(w, 1) for w in tline.split(' ')]

            # if given limit vocabulary
            if self.n_words_src > 0:
                sseq = [w if w < self.n_words_src else 1 for w in sseq]

            # if given limit vocabulary
            if self.n_words_trg > 0:
                tseq = [w if w < self.n_words_trg else 1 for w in tseq]

            # Append sequences to the list
            seqs.append((sseq, tseq))

        sf.close()
        tf.close()

        # Save sequences
        self._seqs = seqs

        # Number of training samples
        self.n_samples = len(self._seqs)

        # Set batch processor function
        self._process_batch = (lambda idxs: self.mask_seqs(idxs))

        if self.shuffle_mode == 'trglen':
            # Homogeneous batches ordered by target sequence length
            # Get an iterator over sample idxs
            self._iter = HomogeneousData(self._seqs, self.batch_size, trg_pos=1)
        else:
            self.rewind()

    def rewind(self):
        if self.shuffle_mode != 'trglen':
            # Fill in the _idxs list for sample order
            if self.shuffle_mode == 'simple':
                # Simple shuffle
                self._idxs = np.random.permutation(self.n_samples).tolist()
            elif self.shuffle_mode is None:
                # Ordered
                self._idxs = np.arange(self.n_samples).tolist()

            self._iter = []
            for i in range(0, self.n_samples, self.batch_size):
                self._iter.append(self._idxs[i:i + self.batch_size])
            self._iter = iter(self._iter)

    def mask_seqs(self, idxs):
        """Prepares a list of padded tensors with their masks for the given sample idxs."""
        src, src_mask = Iterator.mask_data([self._seqs[i][0] for i in idxs])
        trg, trg_mask = Iterator.mask_data([self._seqs[i][1] for i in idxs])
        return (src, src_mask, trg, trg_mask)
