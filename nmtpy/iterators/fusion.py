# -*- coding: utf-8 -*-
import pickle
import numpy as np

from ..sysutils     import listify
from ..nmtutils     import sent_to_idx
from .iterator      import Iterator
from .homogeneous   import HomogeneousData
from ..defaults     import INT, FLOAT

# This is an iterator specifically to be used by the .pkl
# corpora files created for WMT17 Shared Task on Multimodal Machine Translation
# Each element of the list that is pickled is in the following format:
# [src_split_idx, trg_split_idx, imgid, imgname, src_words, trg_words]

# Shorthand for positional access
SSPLIT, TSPLIT, IMGID, IMGNAME, STOKENS, TTOKENS = range(6)

class FusionIterator(Iterator):
    def __init__(self, batch_size, seed=1234, mask=True, shuffle_mode=None, logger=None, **kwargs):
        super(FusionIterator, self).__init__(batch_size, seed, mask, shuffle_mode, logger)

        assert 'pklfile' in kwargs, "Missing argument pklfile"

        # pkl file containing the data
        self.pklfile = kwargs['pklfile']

        # Don't use mask when batch_size == 1 which means we're doing
        # translation with nmt-translate
        if self.batch_size == 1:
            self.mask = False

        # Will be set after reading the data
        self.src_avail = False
        self.trg_avail = False

        # Source word dictionary
        # This may not be available in image captioning
        self.srcdict = kwargs.get('srcdict', None)
        # This may not be available during validation
        self.trgdict = kwargs.get('trgdict', None)

        # Short-list sizes
        self.n_words_src = kwargs.get('n_words_src', 0)
        self.n_words_trg = kwargs.get('n_words_trg', 0)

        # How do we refer to symbolic data variables?
        self.src_name = kwargs.get('src_name', 'x')
        self.trg_name = kwargs.get('trg_name', 'y')

        # Image features file
        #   (n_samples, flattened_spatial, n_maps)
        self.imgfile = kwargs.get('imgfile', None)

        if self.srcdict:
            self._keys = [self.src_name]
            if self.mask:
                self._keys.append("%s_mask" % self.src_name)

        # We have images in the middle
        if self.imgfile:
            self._keys.append("%s_img" % self.src_name)

        if self.trgdict:
            self._keys.append(self.trg_name)
            if self.mask:
                self._keys.append("%s_mask" % self.trg_name)

    def read(self):
        # Load image features file if any
        if self.imgfile is not None:
            self._print('Loading image file...')
            self.img_feats = np.load(self.imgfile)

            # Move n_samples to middle dimension
            # -> 196 x n_samples x 1024 for res4f_relu
            self.img_feats = self.img_feats.transpose(1, 0, 2)

            # (w*h, n, c)
            self.img_shape = tuple((self.img_feats.shape[0], -1, self.img_feats.shape[-1]))
            self._print('Done.')

        # Load the corpora
        with open(self.pklfile, 'rb') as f:
            self._print('Loading pkl file...')
            self._seqs = pickle.load(f)
            self._print('Done.')

        # Introspect the pickle by looking the first sample
        ss = self._seqs[0]

        # we may not have them in pickle or we may not
        # want to use target sentences by giving its vocab None
        if ss[TTOKENS] is not None and self.trgdict:
            self.trg_avail = True

        # Same for source side
        if ss[STOKENS] is not None and self.srcdict:
            self.src_avail = True

        # We now have a list of samples
        self.n_samples = len(self._seqs)

        # Depending on mode, we can have multiple sentences per image so
        # let's store the number of actual images as well.
        # n_unique_samples <= n_samples
        self.n_unique_images = len(set([s[IMGNAME] for s in self._seqs]))

        # Some statistics
        total_src_words = []
        total_trg_words = []

        # Let's map the sentences once to idx's
        for sample in self._seqs:
            if self.src_avail:
                sample[STOKENS] = sent_to_idx(self.srcdict, sample[STOKENS], self.n_words_src)
                total_src_words.extend(sample[STOKENS])
            if self.trg_avail:
                sample[TTOKENS] = sent_to_idx(self.trgdict, sample[TTOKENS], self.n_words_trg)
                total_trg_words.extend(sample[TTOKENS])

        if self.src_avail:
            self.unk_src = total_src_words.count(1)
            self.total_src_words = len(total_src_words)
        if self.trg_avail:
            self.unk_trg = total_trg_words.count(1)
            self.total_trg_words = len(total_trg_words)

        # Set batch processor function
        # idxs can be a list of single element as well
        self._process_batch = (lambda idxs: self.mask_seqs(idxs))

        # Homogeneous batches ordered by target sequence length
        # Get an iterator over sample idxs
        if self.batch_size > 1:
            # Training
            self._iter = HomogeneousData(self._seqs, self.batch_size, trg_pos=TTOKENS)
        else:
            # Test-set
            self._iter = iter([[i] for i in np.arange(self.n_samples)])

    def mask_seqs(self, idxs):
        """Pad if necessary and return padded batches or single samples."""
        data = []

        # Let's fetch batch samples first
        batch = [self._seqs[i] for i in idxs]

        if self.src_avail:
            data += Iterator.mask_data([b[STOKENS] for b in batch], get_mask=self.mask)

        # Source image features
        if self.imgfile is not None:
            x_img = self.img_feats[:, [b[IMGID] for b in batch], :]

            # Reshape accordingly
            x_img.shape = self.img_shape
            data += [x_img]

        if self.trg_avail:
            data += Iterator.mask_data([b[TTOKENS] for b in batch], get_mask=self.mask)

        return data

    def rewind(self):
        # Done automatically within homogeneous iterator
        pass
