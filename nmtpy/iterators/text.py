# -*- coding: utf-8 -*-
import numpy as np

from ..sysutils import fopen
from .iterator import Iterator

"""Text iterator for monolingual data."""
class TextIterator(Iterator):
    def __init__(self, batch_size, seed=1234, mask=True, shuffle_mode=None, logger=None, **kwargs):
        super(TextIterator, self).__init__(batch_size, seed, mask, shuffle_mode, logger)

        assert 'file'   in kwargs, "Missing argument file"
        assert 'dict'   in kwargs, "Missing argument dict"

        self.__file = kwargs['file']
        self.__dict = kwargs['dict']
        self.__n_words = kwargs.get('n_words', 0)
        self.name = kwargs.get('name', 'x')

        self._keys = [self.name]
        if self.mask:
            self._keys.append('%s_mask' % self.name)

    def read(self):
        seqs = []
        with fopen(self.__file, 'r') as f:
            for idx, line in enumerate(f):
                line = line.strip()

                # Skip empty lines
                if line == "":
                    print('Warning: empty line in %s' % self.__file)
                else:
                    line = line.split(" ")

                    seq = [self.__dict.get(w, 1) for w in line]

                    # if given limit vocabulary
                    if self.__n_words > 0:
                        seq = [w if w < self.__n_words else 1 for w in seq]
                    # Append the sequence
                    seqs += [seq]

        self._seqs = seqs
        self.n_samples = len(self._seqs)
        self._idxs = np.arange(self.n_samples)

        if not self._minibatches:
            self.prepare_batches()
        self.rewind()

    def prepare_batches(self):
        self._minibatches = []

        for i in range(0, self.n_samples, self.batch_size):
            batch_idxs = self._idxs[i:i + self.batch_size]
            x, x_mask = Iterator.mask_data([self._seqs[i] for i in batch_idxs])
            self._minibatches.append((x, x_mask))

    def rewind(self):
        """Recreate the iterator."""
        if self.shuffle_mode == 'simple':
            self._idxs = np.random.permutation(self.n_samples)
            self.prepare_batches()

        self._iter = iter(self._minibatches)
