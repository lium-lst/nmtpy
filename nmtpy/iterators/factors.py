# -*- coding: utf-8 -*-
import numpy as np

from ..sysutils   import fopen
from .iterator    import Iterator
from .homogeneous import HomogeneousData
from ..defaults import INT, FLOAT

#from six.moves import range
#from six.moves import zip

#from collections import OrderedDict

#from ..nmtutils import sent_to_idx
#from ..typedef import *

"""Parallel text iterator for translation data."""
class FactorsIterator(Iterator):
    def __init__(self, batch_size, seed=1234, mask=True, shuffle_mode=None, **kwargs):
        super(FactorsIterator, self).__init__(batch_size, seed, mask, shuffle_mode)

        
        #TODO add pkl files reading
        # TODO add dict with files and pass it as argument

        # How do we use the multimodal data?
        # 'all'     : All combinations (~725K parallel)
        # 'single'  : Take only the first pair e.g., train0.en->train0.de (~29K parallel)
        # 'pairs'   : Take only one-to-one pairs e.g., train_i.en->train_i.de (~145K parallel)
        self.mode = kwargs.get('mode', 'all')
        self._print('Shuffle mode: %s' % shuffle_mode)
        
        self.srcfile = kwargs['srcfile']
        self.srcdict = kwargs['srcdict']

        # 2 input source
        if 'srcfactfile' in kwargs:
            self.srcfact = True
            self.trgfact = True
            
            self.srcfactfile = kwargs['srcfactfile']
            self.srcfactdict = kwargs['srcfactdict']
        
            self.n_words_src = kwargs.get('n_words_src', 0)
            self.n_words_srcfact = kwargs.get('n_words_srcfact', 0)
        
            self.src_name = kwargs.get('src_name', 'x1')
            self.srcfact_name = kwargs.get('srcfact_name', 'x2')
            self._keys = [self.src_name]
            self._keys.append(self.srcfact_name)
            if self.mask:
                self._keys.append("%s_mask" % self.src_name)
                self._keys.append("%s_mask" % self.srcfact_name)
        # 1 input source
        else:
            self.srcfact = False
            
            self.n_words_src = kwargs.get('n_words_src', 0)
            self.src_name = kwargs.get('src_name', 'x')
            self._keys = [self.src_name]
            if self.mask:
                self._keys.append("%s_mask" % self.src_name)

        # 2 output target
        if 'trgfactfile' in kwargs:
            self.trgfact = True
            self.trglemfile = kwargs['trglemfile']
            self.trgfactfile = kwargs['trgfactfile']

            self.trglemdict = kwargs['trglemdict']
            self.trgfactdict = kwargs['trgfactdict']

            self.n_words_trglem = kwargs.get('n_words_trglem', 0)
            self.n_words_trgfact = kwargs.get('n_words_trgfact', 0)

            self.trglem_name = kwargs.get('trglem_name', 'y1')
            self.trgfact_name = kwargs.get('trgfact_name', 'y2')

            self._keys.append(self.trglem_name)
            self._keys.append(self.trgfact_name)
            if self.mask:
                self._keys.append("%s_mask" % self.trglem_name)
                self._keys.append("%s_mask" % self.trgfact_name)
    
        # 1 output target
        else:
            self.trgfact = False
            self.trgfile = kwargs['trgfile']
            self.trgdict = kwargs['trgdict']
            self.n_words_trg = kwargs.get('n_words_trg', 0)
            self.trg_name = kwargs.get('trg_name', 'y')
            self._keys.append(self.trg_name)
            if self.mask:
                self._keys.append("%s_mask" % self.trg_name)

    def read(self):
        seqs = []
        # 2 inputs and 2 outputs
        if self.srcfact and self.trgfact:
            tlf = fopen(self.trglemfile, 'r')
            tff = fopen(self.trgfactfile, 'r')
            slf = fopen(self.srcfile, 'r')
            sff = fopen(self.srcfactfile, 'r')

            for idx, (slline, sfline, tlline, tfline) in enumerate(zip(slf, sff, tlf, tff)):
                slline = slline.strip()
                sfline = sfline.strip()
                tlline = tlline.strip()
                tfline = tfline.strip()
        
                # Exception if empty line found
                if slline == "" or sfline == "" or tlline == "" or tfline == "":
                    continue
            
                slseq = [self.srcdict.get(w, 1) for w in slline.split(' ')]
                sfseq = [self.srcfactdict.get(w, 1) for w in sfline.split(' ')]
                tlseq = [self.trglemdict.get(w, 1) for w in tlline.split(' ')]
                tfseq = [self.trgfactdict.get(w, 1) for w in tfline.split(' ')]
            
                # if given limit vocabulary
                if self.n_words_src > 0:
                    slseq = [w if w < self.n_words_src else 1 for w in slseq]
                if self.n_words_srcfact > 0:
                    sfseq = [w if w < self.n_words_srcfact else 1 for w in sfseq]
                if self.n_words_trglem > 0:
                    tlseq = [w if w < self.n_words_trglem else 1 for w in tlseq]
                if self.n_words_trgfact > 0:
                    tfseq = [w if w < self.n_words_trgfact else 1 for w in tfseq]
        
                # Append sequences to the list
                seqs.append((slseq, sfseq, tlseq, tfseq))

            slf.close()
            sff.close()
            tlf.close()
            tff.close()

        # 2 inputs and 1 output
        elif self.srcfact:
            slf = fopen(self.srcfile, 'r')
            sff = fopen(self.srcfactfile, 'r')
            tf = fopen(self.trgfile, 'r')

            for idx, (slline, sfline, tline) in enumerate(zip(slf, sff, tf)):
                slline = slline.strip()
                sfline = sfline.strip()
                tline = tline.strip()
        
                # Exception if empty line found
                if slline == "" or sfline == "" or tline == "":
                    continue
            
                slseq = [self.srcdict.get(w, 1) for w in slline.split(' ')]
                sfseq = [self.srcfactdict.get(w, 1) for w in sfline.split(' ')]
                tseq = [self.trgdict.get(w, 1) for w in tline.split(' ')]
            
                # if given limit vocabulary
                if self.n_words_src > 0:
                    slseq = [w if w < self.n_words_src else 1 for w in slseq]
                if self.n_words_srcfact > 0:
                    sfseq = [w if w < self.n_words_srcfact else 1 for w in sfseq]
                if self.n_words_trg > 0:
                    tfseq = [w if w < self.n_words_trg else 1 for w in tseq]
        
                # Append sequences to the list
                seqs.append((slseq, sfseq, tseq))

            slf.close()
            sff.close()
            tf.close()

        # 1 input and 2 outputs
        elif self.trgfact:
            # We open the data files
            sf = fopen(self.srcfile, 'r')
            tlf = fopen(self.trglemfile, 'r')
            tff = fopen(self.trgfactfile, 'r')
            # We iterate the data files
            for idx, (sline, tlline, tfline) in enumerate(zip(sf, tlf, tff)):
                sline = sline.strip()
                tlline = tlline.strip()
                tfline = tfline.strip()

                # Exception if empty line found
                if sline == "" or tlline == "" or tfline == "":
                    continue
                # For each word in the sentence we add its corresponding ID in the dic
                seq = [self.srcdict.get(w, 1) for w in sline.split(' ')]
                lseq = [self.trglemdict.get(w, 1) for w in tlline.split(' ')]
                fseq = [self.trgfactdict.get(w, 1) for w in tfline.split(' ')]

                #if given limit vocabulary
                if self.n_words_src > 0:
                    sseq = [w if w < self.n_words_src else 1 for w in seq]

                # if given limit vocabulary
                if self.n_words_trglem > 0:
                    tlseq = [w if w < self.n_words_trglem else 1 for w in lseq]
                if self.n_words_trgfact > 0:
                    tfseq = [w if w < self.n_words_trgfact else 1 for w in fseq]

                # Append sequences to the list
                seqs.append((sseq, tlseq, tfseq))
        
            sf.close()
            tlf.close()
            tff.close()

        # Save sequences
        self._seqs = seqs

        # Number of training samples
        self.n_samples = len(self._seqs)
        # TODO statistics

        if self.shuffle_mode == 'trglen':
            # Homogeneous batches ordered by target sequence length
            # Get an iterator over sample idxs
            self._iter = HomogeneousData(self._seqs, self.batch_size, trg_pos=1)
            self._process_batch = (lambda idxs: self.mask_seqs(idxs))
        else:
            if self.shuffle_mode == 'simple':
                # Simple shuffle
                self._idxs = np.random.permutation(self.n_samples)
            else:
                # Ordered
                self._idxs = np.arange(self.n_samples)
            self.prepare_batches()


    # this method is required for the 2 masks for target because we do not want EOS in the second output
    @staticmethod
    def mask_data_mult(seqs):
        """Pads sequences with EOS (0) for minibatch processing."""
        lengths = [len(s) for s in seqs]
        n_samples = len(seqs)
     
        maxlen = np.max(lengths) + 1
     
        # Shape is (t_steps, samples)
        x = np.zeros((maxlen, n_samples)).astype(INT)
        x_mask = np.zeros_like(x).astype(FLOAT)
     
        for idx, s_x in enumerate(seqs):

            x[:lengths[idx], idx] = s_x
            x_mask[:lengths[idx], idx] = 1.
        
        return x, x_mask


    def mask_seqs(self, idxs):
        """Prepares a list of padded tensors with their masks for the given sample idxs."""
        src, src_mask = Iterator.mask_data([self._seqs[i][0] for i in idxs])
        if self.srcfact and self.trgfact:
            srcfact, srcmult_mask = Iterator.mask_data([self._seqs[i][1] for i in idxs])
            trg, trg_mask = Iterator.mask_data([self._seqs[i][2] for i in idxs])
            trgmult, trgmult_mask = Iterator.mask_data([self._seqs[i][3] for i in idxs])
            return (src, srcfact, src_mask, trg, trgmult, trg_mask)
        elif self.srcfact:
            srcfact, srcmult_mask = Iterator.mask_data([self._seqs[i][1] for i in idxs])
            trg, trg_mask = Iterator.mask_data([self._seqs[i][2] for i in idxs])
            return (src, srcfact, src_mask, srcmult_mask, trg, trg_mask)
        elif self.trgfact:
            trg, trg_mask = Iterator.mask_data([self._seqs[i][1] for i in idxs])
            trgmult, trgmult_mask = self.mask_data_mult([self._seqs[i][2] for i in idxs])
            return (src, src_mask, trg, trgmult, trg_mask, trgmult_mask)

    def prepare_batches(self):
        self._minibatches = []

        for i in range(0, self.n_samples, self.batch_size):
            batch_idxs = self._idxs[i:i + self.batch_size]
            self._minibatches.append(self.mask_seqs(batch_idxs))

        self.rewind()

    def rewind(self):
        if self.shuffle_mode != 'trglen':
            self._iter = iter(self._minibatches)
