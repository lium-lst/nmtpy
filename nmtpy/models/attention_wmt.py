# -*- coding: utf-8 -*-
from ..iterators.fusion import FusionIterator
from .attention import Model as Attention

# Same model as attention but using FusionIterator
# Purpose was to train a monomodal system using the same .pkl
# files prepared for multimodal Task 2 system.
#
# FIXME: Not tested since WMT16 Task 2 experiments, probably broken
class Model(Attention):
    def __init__(self, seed, logger, **kwargs):
        # Call parent's init first
        super(Model, self).__init__(seed, logger, **kwargs)

        self.data_mode = kwargs.pop('data_mode', 'pairs')

    def load_valid_data(self, from_translate=False, data_mode='single'):
        if from_translate:
            self.valid_ref_files = self.data['valid_trg']
            if isinstance(self.valid_ref_files, str):
                self.valid_ref_files = list([self.valid_ref_files])

            self.valid_iterator = FusionIterator(
                    mask=False,
                    batch_size=1,
                    pklfile=self.data['valid_src'],
                    srcdict=self.src_dict, n_words_src=self.n_words_src,
                    mode=data_mode)
        else:
            # Take the first validation item for NLL computation
            self.valid_iterator = FusionIterator(
                    batch_size=self.batch_size,
                    pklfile=self.data['valid_src'],
                    trgdict=self.trg_dict, srcdict=self.src_dict,
                    n_words_trg=self.n_words_trg, n_words_src=self.n_words_src,
                    mode='single') # Override the given parameter

        self.valid_iterator.read()

    def load_data(self):
        self.train_iterator = FusionIterator(
                batch_size=self.batch_size,
                shuffle_mode=self.smode,
                logger=self.__logger,
                pklfile=self.data['train_src'],
                trgdict=self.trg_dict, srcdict=self.src_dict,
                n_words_trg=self.n_words_trg, n_words_src=self.n_words_src,
                mode=self.data_mode)
        # Prepare batches
        self.train_iterator.read()
        self.load_valid_data()
