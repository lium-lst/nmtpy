#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

# Avoid thread explosion
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys
import argparse
import platform
import textwrap
import importlib

from nmtpy.config import Config
from nmtpy.logger import Logger
from nmtpy.sysutils import *
from nmtpy.nmtutils import get_param_dict
from nmtpy.mainloop import MainLoop

# Ensure cleaning up temp files and processes
import nmtpy.cleanup as cleanup

# Import defaults
from nmtpy.defaults import TRAIN_DEFAULTS as trdefs
from nmtpy.defaults import MODEL_DEFAULTS as mddefs

if __name__ == '__main__':
    # Pretty print defaults
    defs  = '\n' + pretty_dict(trdefs, 'Training defaults') + '\n\n'
    defs += pretty_dict(mddefs, 'Model defaults')

    parser = argparse.ArgumentParser(prog='nmt-train',
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=textwrap.dedent('''\
                                        nmt-train trains the given model on a free GPU.
                                        All the details regarding the model and the hyper-parameters
                                        are given through -c/--config flag.

                                        A list of sane defaults are defined in nmtpy/defaults.py and
                                        listed below. These defaults are used if you did not override
                                        them in the configuration file.

                                        A final way of overriding parameters are through the variable
                                        length 'extra' arguments.

                                        Example:
                                          $ nmt-train -c sample.conf
                                          # Change seed and model_type by overriding them
                                          $ nmt-train -c sample.conf "seed:1235" "model_type:att"
                                          ''') + defs,
                                     argument_default=argparse.SUPPRESS)

    # Mandatory argument pointing to the configuration file
    parser.add_argument('-c', '--config'        , help="Path to model configuration file",
                                                  type=str, required=True)

    # Override the model-type given in configuration file to experiment with
    # a different model by fixing every other parameter given in the configuration
    parser.add_argument('-m', '--model-type'    , help="Override the model type given in the configuration",
                                                  type=str)

    parser.add_argument('-s', '--suffix'        , help="Model file suffix",
                                                  type=str, default=None)
    parser.add_argument('-i', '--init'          , help="Pretrained weights .npz extracted with nmt-extract",
                                                  type=str)
    parser.add_argument('-f', '--freeze'        , help="Freeze the pretrained weights given with --init",
                                                  action="store_true", default=False)
    parser.add_argument('-t', '--timestamp'     , help="Add timestamp to log messages.",
                                                  action="store_true", default=False)
    parser.add_argument('-n', '--no-log'        , help="Do not log to text file.",
                                                  action="store_true", default=False)

    parser.add_argument('-v', '--verbose'       , help="Dump Theano graph and inspect optimization.",
                                                  action="store_true", default=False)

    # You can basically override everything by passing 'lrate: 0.1' style strings at the end
    # of command-line arguments
    parser.add_argument('extra'                 , help="List of 'key:value' to override configuration",
                                                  nargs="*", default=[])

    ####################################
    # Parse command-line arguments first
    ####################################
    cargs   = parser.parse_args()

    # Pop config filename, verbose flag and extra
    cfname  = cargs.__dict__.pop('config')
    verbose = cargs.__dict__.pop('verbose')
    extra   = cargs.__dict__.pop('extra')
    suffix  = cargs.__dict__.pop('suffix')
    tstamp  = cargs.__dict__.pop('timestamp')
    nolog   = cargs.__dict__.pop('no_log')
    freeze  = cargs.__dict__.pop('freeze')

    # Take the remaining command line arguments (model_type and/or init if any)
    cmd_args = cargs.__dict__

    if len(extra) > 0:
        # Split and convert extra arguments to dict
        ext_args = [e.split(':', 1) for e in extra]
        ext_args = dict([(k.strip(), v.strip()) for k,v in ext_args])

        # Merge by prioritizing extra arguments
        cmd_args.update(ext_args)

    # Parse configuration file and merge with the rest
    conf = Config(cfname, trdefs=trdefs, mddefs=mddefs, override=cmd_args)
    train_args, model_args = conf.parse()

    # Create a folder named as conf file
    folder_name = os.path.splitext(os.path.basename(cfname))[0]
    model_args.save_path = os.path.join(model_args.save_path, folder_name)
    ensure_dirs([model_args.save_path])

    # Create a unique experience identifier string
    exp_id = get_exp_identifier(train_args, model_args, suffix=suffix)
    # Get unique run identifier (starts from 1)
    run_id = get_next_runid(model_args.save_path, exp_id)
    # Get log file name
    log_fname = None
    if not nolog:
        log_fname = os.path.join(model_args.save_path,
                                "%s.%d.log" % (exp_id, run_id))

    # Start logging module (both to terminal and to file)
    Logger.setup(log_file=log_fname, timestamp=tstamp)
    log = Logger.get()
    cleanup.register_handler(log)

    # Update save_path
    model_args.save_path = os.path.join(model_args.save_path,
                                        "%s.%d" % (exp_id, run_id))

    # ensure valid hyps folder if valid_save_hyp is activated
    if train_args.valid_save_hyp is True:
        ensure_dirs([model_args.save_path+'.valid_hyps'])

    ###################################
    # Set device for Theano
    if 'THEANO_FLAGS' not in os.environ:
        train_args.device_id = get_device(train_args.device_id)

        # Check for GPUARRAY to switch to new Theano backend
        if train_args.device_id.startswith('gpu') and "GPUARRAY" in os.environ:
            train_args.device_id = train_args.device_id.replace('gpu', 'cuda')

        os.environ['THEANO_FLAGS'] = "device=%s" % train_args.device_id

    log.info("THEANO_FLAGS = %s" % os.environ['THEANO_FLAGS'])

    # Import theano
    import theano
    import numpy as np
    log.info("Using device: %s (on machine %s)" % (train_args.device_id, platform.node()))
    log.info("Theano version: %s" % theano.version.full_version)

    # Set numpy random seed before everything else
    if train_args.seed != 0:
        np.random.seed(train_args.seed)

    # Print options
    print_summary(train_args, model_args, print_func=log.info)

    # Import the model
    Model = importlib.import_module("nmtpy.models.%s" % train_args.model_type).Model

    # Create model object
    # Save model_type into the model as well
    model = Model(seed=train_args.seed, logger=log,
                  model_type=train_args.model_type, **(model_args.__dict__))

    # Initialize parameters
    log.info("Initializing parameters")
    model.init_params()

    # Create theano shared variables
    log.info('Creating shared variables')
    model.init_shared_variables()

    # List of weights that will not receive updates during BP
    dont_update = []

    # Override some weights with pre-trained ones if given
    if train_args.init:
        log.info('Will override parameters from pre-trained weights')
        log.info('  %s' % os.path.basename(train_args.init))
        new_params = get_param_dict(train_args.init)
        model.update_shared_variables(new_params)
        if freeze:
            log.info('Pretrained weights will not be updated.')
            dont_update = list(new_params.keys())

    # Print number of parameters
    log.info("Number of parameters: %s" % model.get_nb_params())

    # Load data
    log.info("Loading data")
    model.load_data()

    # Dump model information
    model.info()

    # Build the model
    log.info("Building model")
    data_loss = model.build()

    log.info("Input tensor order: ")
    log.info(list(model.inputs.values()))

    if train_args.sample_freq > 0:
        log.info('Building sampler')
        model.build_sampler()

    # Compute regularized loss
    reg_loss = []
    if train_args.decay_c > 0:
        reg_loss.append(model.get_l2_weight_decay(train_args.decay_c))

    reg_loss = sum(reg_loss) if len(reg_loss) > 0 else None

    # Build optimizer
    log.info('Building optimizer %s (initial lr=%.5f)' % (model_args.optimizer, model_args.lrate))
    model.build_optimizer(data_loss, reg_loss, train_args.clip_c, dont_update=dont_update, debug=verbose)

    # Save graph
    if verbose:
        theano.printing.debugprint(model.train_batch, file=open('%s.graph' % log_file.replace(".log", ""), 'w'))

    # Reseed to retain the order of shuffle operations
    np.random.seed(train_args.seed)

    # Create mainloop
    loop = MainLoop(model, log, train_args, model_args)
    loop.run()
