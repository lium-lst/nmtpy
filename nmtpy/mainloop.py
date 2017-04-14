# -*- coding: utf-8 -*-
from collections import OrderedDict
from shutil import copy

from nmtpy.metrics import is_last_best, find_best, comparators
from nmtpy.sysutils import force_symlink

import numpy as np
import time
import os

class MainLoop(object):
    def __init__(self, model, logger, train_args, model_args):
        # NOTE: model_args not used, if necessary they should be accessible
        # from self.model.*

        self.model          = model                         # The model instance that is trained
        self.__log          = logger                        # logger instance

        # Counters
        self.uctr           = 0                             # update ctr
        self.ectr           = 0                             # epoch ctr
        self.vctr           = 0                             # validation ctr
        self.early_bad      = 0                             # early-stop counter

        self.f_snapshot     = train_args.snapshot_freq      # TODO: Modify this for periodic saving
        self.save_best_n    = train_args.save_best_n        # Keep N best validation models on disk
        self.max_updates    = train_args.max_iteration      # Training stops if uctr hits 'max_updates'
        self.max_epochs     = train_args.max_epochs         # Training stops if ectr hits 'max_epochs'

        # Validation related parameters
        self.patience       = train_args.patience           # Stop training if no improvement after this validations
        self.valid_start    = train_args.valid_start        # Start validation at epoch 'valid_start'
        self.beam_size      = train_args.valid_beam         # Beam size for validation decodings
        self.njobs          = train_args.valid_njobs        # # of CPU processes for validation decodings
        self.f_valid        = train_args.valid_freq         # Validation frequency in terms of updates
        self.epoch_valid    = (self.f_valid == 0)           # 0: end of epochs
        self.valid_save_hyp = train_args.valid_save_hyp     # save validation hypotheses under 'valid_hyps' folder
        self.f_verbose      = 10                            # Print frequency

        # TODO: Remove sampling stuff, not useful
        self.f_sample       = train_args.sample_freq
        self.do_sampling    = self.f_sample > 0
        self.n_samples      = 5                             # Number of samples to produce

        self.epoch_losses   = []

        # Multiple comma separated metrics are supported
        # Each key is a metric name, values are metrics so far.
        self.valid_metrics = OrderedDict()

        # We may have no validation data.
        if self.f_valid >= 0:
            # NOTE: This is relevant only for fusion models + WMTIterator
            self.valid_mode = 'single'
            if 'valid_mode' in self.model.__dict__:
                self.valid_mode = self.model.valid_mode

            # Setup validation hypotheses folder name
            if self.valid_save_hyp:
                base_folder = self.model.save_path + '.valid_hyps'
                self.valid_save_prefix = os.path.join(base_folder, os.path.basename(self.model.save_path))

            # Requested metrics, replace px with loss
            metrics = train_args.valid_metric.replace('px', 'loss').split(',')

            # first one is for early-stopping
            self.early_metric = metrics[0]

            for metric in metrics:
                self.valid_metrics[metric] = []

            # Ensure that loss exists
            self.valid_metrics['loss'] = []

            # Prepare the string to pass to beam_search
            self.beam_metrics = ",".join([m for m in self.valid_metrics if m != 'loss'])

            # Best N checkpoint saver
            self.best_models = []

        # FIXME: Disable TB support for now
        self.__tb = None
        ####################
        ## Tensorboard setup
        ####################
#        try:
            #from tensorboard import SummaryWriter
        #except ImportError as ie:
            #self.__tb = None
        #else:
            ## FIXME: This should be a global folder with subfolder
            ## for each training so that we can follow many systems on TB
            #self.__print('Will log training progress to TensorBoard')
            #self.__tb = SummaryWriter(os.path.dirname(self.model.save_path))

    def __send_stats(self, step, **kwargs):
        """Send statistics to TensorBoard."""
        if self.__tb:
            for name, value in kwargs.items():
                if isinstance(value, tuple):
                    # Metric tuple, pass float value
                    value = value[-1]
                self.__tb.add_scalar(name, value, global_step=step)

    def __print(self, msg, footer=False):
        """Pretty prints message with optional separator."""
        self.__log.info(msg)
        if footer:
            self.__log.info('-' * len(msg))

    def __save_best_model(self):
        """Saves best N models to disk."""
        if self.save_best_n > 0:
            # Get the score of the system that will be saved
            cur_score = self.valid_metrics[self.early_metric][-1]

            # Custom filename with metric score
            cur_fname = "%s-val%3.3d-%s_%.3f.npz" % (self.model.save_path, self.vctr, self.early_metric, cur_score)

            # Stack is empty, save the model whatsoever
            if len(self.best_models) < self.save_best_n:
                self.best_models.append((cur_score, cur_fname))

            # Stack is full, replace the worst model
            else:
                os.unlink(self.best_models[self.next_prune_idx][1])
                self.best_models[self.next_prune_idx] = (cur_score, cur_fname)

            self.__print('Saving model with best validation %s' % self.early_metric.upper())
            self.model.save(cur_fname)

            # Create a .BEST symlink
            force_symlink(cur_fname, ('%s.BEST.npz' % self.model.save_path), relative=True)

            # In the next best, we'll remove the following idx from the list/disk
            # Metric specific comparator stuff
            where = comparators[self.early_metric][-1]
            self.next_prune_idx = sorted(range(len(self.best_models)),
                                         key=self.best_models.__getitem__)[where]

    def __update_lrate(self):
        """Update learning rate by annealing it."""
        pass

    def __train_epoch(self):
        """Train a full epoch."""
        self.ectr += 1

        start = time.time()
        start_uctr = self.uctr
        self.__print('Starting Epoch %d' % self.ectr, True)

        batch_losses = []

        # Iterate over batches
        for data in self.model.train_iterator:
            self.uctr += 1

            # Forward/backward and get loss
            loss = self.model.train_batch(*list(data.values()))
            batch_losses.append(loss)
            self.__send_stats(self.uctr, train_loss=loss)

            # Verbose
            if self.uctr % self.f_verbose == 0:
                self.__print("Epoch: %6d, update: %7d, cost: %10.6f" % (self.ectr, self.uctr, loss))

            # Should we stop
            if self.uctr == self.max_updates:
                self.__print("Max iteration %d reached." % self.uctr)
                return False

            # Update learning rate if requested
            self.__update_lrate()

            # Do sampling
            self.__do_sampling(data)

            # Do validation
            if not self.epoch_valid and self.f_valid > 0 and self.uctr % self.f_valid == 0:
                self.__do_validation()

            # Check stopping conditions
            if self.early_bad == self.patience:
                self.__print("Early stopped.")
                return False

        # An epoch is finished
        epoch_time = time.time() - start

        # Print epoch summary
        up_ctr = self.uctr - start_uctr
        self.__dump_epoch_summary(batch_losses, epoch_time, up_ctr)

        # Do validation
        if self.epoch_valid:
            self.__do_validation()

        # Check whether maximum epoch is reached
        if self.ectr == self.max_epochs:
            self.__print("Max epochs %d reached." % self.max_epochs)
            return False

        return True

    def __do_sampling(self, data):
        """Generates samples and prints them."""
        if self.do_sampling and self.uctr % self.f_sample == 0:
            samples = self.model.generate_samples(data, self.n_samples)
            if samples is not None:
                for src, truth, sample in samples:
                    if src:
                        self.__print("Source: %s" % src)
                    self.__print.info("Sample: %s" % sample)
                    self.__print.info(" Truth: %s" % truth)

    def __do_validation(self):
        """Do early-stopping validation."""
        if self.ectr >= self.valid_start:
            self.vctr += 1

            # Compute validation loss
            self.model.set_dropout(False)
            cur_loss = self.model.val_loss()
            self.model.set_dropout(True)

            # Add val_loss
            self.valid_metrics['loss'].append(cur_loss)

            # Print validation loss
            self.__print("Validation %2d - LOSS = %.3f (PPL: %.3f)" % (self.vctr, cur_loss, np.exp(cur_loss)))

            #############################
            # Are we doing beam search? #
            #############################
            if self.beam_metrics:
                beam_results = None
                # Save beam search results?
                f_valid_out = None

                if self.valid_save_hyp:
                    f_valid_out = "{0}.{1:03d}".format(self.valid_save_prefix, self.vctr)

                self.__print('Calling beam-search process')
                beam_time = time.time()
                beam_results = self.model.run_beam_search(beam_size=self.beam_size,
                                                          n_jobs=self.njobs,
                                                          metric=self.beam_metrics,
                                                          mode='beamsearch',
                                                          valid_mode=self.valid_mode,
                                                          f_valid_out=f_valid_out)
                beam_time = time.time() - beam_time
                self.__print('Beam-search ended, took %.5f minutes.' % (beam_time / 60.))

                if beam_results:
                    # beam_results: {name: (metric_str, metric_float)}
                    # names are as defined in metrics/*.py like BLEU, METEOR
                    # but we use lowercase names in conf files.
                    self.__send_stats(self.vctr, **beam_results)
                    for name, (metric_str, metric_value) in beam_results.items():
                        self.__print("Validation %2d - %s" % (self.vctr, metric_str))
                        self.valid_metrics[name.lower()].append(metric_value)
                else:
                    self.__print('Skipping this validation since nmt-translate probably failed.')
                    # Return back to training loop since nmt-translate did not run correctly.
                    # This will allow to fix your model's build_sampler while training continues.
                    return

            # Is this the best evaluation based on early-stop metric?
            if is_last_best(self.early_metric, self.valid_metrics[self.early_metric]):
                if self.valid_save_hyp:
                    # Create a link towards best hypothesis file
                    force_symlink(f_valid_out, '%s.BEST' % self.valid_save_prefix, relative=True)

                self.__save_best_model()
                self.early_bad = 0
            else:
                self.early_bad += 1
                self.__print("Early stopping patience: %d validation left" % (self.patience - self.early_bad))

            self.__dump_val_summary()

    def __dump_val_summary(self):
        """Print validation summary."""
        for metric, history in self.valid_metrics.items():
            if len(history) > 0:
                # Find the best validation idx and value so far
                best_idx, best_val = find_best(metric, history)
                if metric == 'loss':
                    msg = "BEST %s = %.3f (PPL: %.3f)" % (metric.upper(), best_val, np.exp(best_val))
                else:
                    msg = "BEST %s = %.3f" % (metric.upper(), best_val)

                self.__print('--> Current %s at validation %d' % (msg, best_idx))

        # Remember who we are
        self.__print('--> This is model: %s' % os.path.basename(self.model.save_path))

    def __dump_epoch_summary(self, losses, epoch_time, up_ctr):
        """Print epoch summary."""
        update_time = epoch_time / float(up_ctr)
        mean_loss = np.array(losses).mean()
        self.epoch_losses.append(mean_loss)

        self.__print("--> Epoch %d finished with mean loss %.5f (PPL: %4.5f)" % (self.ectr, mean_loss, np.exp(mean_loss)))
        self.__print("--> Epoch took %.3f minutes, %.3f sec/update" % ((epoch_time / 60.0), update_time))

    def run(self):
        """Run training loop."""
        self.model.set_dropout(True)
        #self.model.save(self.model.save_path + '.npz')
        while self.__train_epoch():
            pass

        # Final summary
        if self.f_valid >= 0:
            self.__dump_val_summary()
        else:
            # No validation data used, save the final model
            self.__print('Saving final model.')
            self.model.save("%s.npz" % self.model.save_path)
