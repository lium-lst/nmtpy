# -*- coding: utf-8 -*-
import random

from abc import ABCMeta, abstractmethod
from collections import OrderedDict

import numpy as np
from ..defaults import INT, FLOAT

class Iterator(object, metaclass=ABCMeta):
    """Base Iterator class."""

    @staticmethod
    def mask_data(seqs, get_mask=True):
        """Pads sequences with EOS (0) for minibatch processing."""
        lengths = [len(s) for s in seqs]
        maxlen = np.max(lengths) + 1

        # Shape is (t_steps, samples)
        x = np.zeros((maxlen, len(seqs))).astype(INT)
        x_mask = np.zeros_like(x).astype(FLOAT)

        for idx, s_x in enumerate(seqs):
            x[:lengths[idx], idx] = s_x
            x_mask[:lengths[idx] + 1, idx] = 1.

        if get_mask:
            return [x, x_mask]
        else:
            return [x]

    def _print(self, msg):
        if self.logger:
            self.logger.info(msg)

    def __init__(self, batch_size, seed=1234, mask=True, shuffle_mode=None, logger=None):
        self.n_samples  = 0
        self.seed       = seed
        self.mask       = mask
        self.logger     = logger
        self.batch_size = batch_size
        self._keys     = []
        self._idxs     = []
        self._seqs     = []
        self._iter     = None
        self._minibatches = []

        self.shuffle_mode = shuffle_mode
        if self.shuffle_mode:
            # Set random seed
            random.seed(self.seed)

        # This can be set by child classes for processing
        # a list of idxs into the actual minibatch
        self._process_batch = lambda x: x

    def __len__(self):
        """Returns number of samples."""
        return self.n_samples

    def __iter__(self):
        return self

    def __next__(self):
        """Returns the next set of data from the iterator."""
        try:
            data = self._process_batch(next(self._iter))
        except StopIteration as si:
            self.rewind()
            raise
        else:
            # Lookup the keys and return an ordered dict of the current minibatch
            return OrderedDict([(k, data[i]) for i,k in enumerate(self._keys)])

    # May or may not be used.
    def prepare_batches(self):
        """Prepare self.__iter."""
        pass

    @abstractmethod
    def read(self):
        """Read the data and put in into self.__seqs."""
        pass

    @abstractmethod
    def rewind(self):
        pass
