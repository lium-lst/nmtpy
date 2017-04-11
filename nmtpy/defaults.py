# -*- coding: utf-8 -*-

# Default data types
INT   = 'int64'
FLOAT = 'float32'

MODEL_DEFAULTS = {
        'weight_init':        'xavier',       # Can be a float for the scale of normal initialization, "xavier" or "he".
        'batch_size':         32,             # Training batch size
        'optimizer':          'adam',         # adadelta, sgd, rmsprop, adam
        'lrate':              0.0004,         # Initial learning rate
        }

TRAIN_DEFAULTS = {
        'init':               None,           # Pretrained model .npz file
        'device_id':          'auto',         #
        'seed':               1234,           # RNG seed
        'clip_c':             5.,             # Clip gradients above clip_c
        'decay_c':            0.,             # L2 penalty factor
        'patience':           10,             # Early stopping patience
        'max_epochs':         100,            # Max number of epochs to train
        'max_iteration':      int(1e6),       # Max number of updates to train
        'valid_metric':       'bleu',         # one or more metrics separated by comma, 1st one used for early-stopping
        'valid_start':        1,              # Epoch which validation will start
        'valid_njobs':        16,             # # of parallel CPU tasks to do beam-search
        'valid_beam':         12,             # Allow changing beam size during validation
        'valid_freq':         0,              # 0: End of epochs
        'valid_save_hyp':     False,          # Save each output of validation to separate files
        'sample_freq':        0,              # Sampling frequency during training (0: disabled)
        'snapshot_freq':      0,              # Checkpoint frequency in terms of number of iterations
        'save_best_n':        4,              # Always keep a set of 4 best validation models on disk
        }
