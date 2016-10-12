#!/usr/bin/env python
import os
import sys
import time
import uuid
import random
import argparse
import subprocess
import xml.etree.ElementTree as ET

from collections import OrderedDict
from threading import Semaphore

import numpy as np

######################################
# Will be initialized from inside main
INTERRUPTED     = False
SEMAPHORE_GPU   = None
PROCS           = OrderedDict()
DEVNULL         = open(os.devnull, 'w')
N_GPU           = 0

SEEDS           = [1234]

# Can try different architectures with same parameter sets
ARCHS           = ["attentionv2", "attentionv4"]

# Try equal dims for emb and rnn [60,200] with step=10
DIMS            = range(60, 210, 10)

# Will be prefilled before launching jobs
EXPERIMENTS     = []
######################################

##################
# Helper functions
##################
def get_gpu_count():
    """Returns GPU count on the machine."""
    root = ET.fromstring(subprocess.check_output(['nvidia-smi', '-x', '-q']))
    return len(root.findall('gpu'))

def get_empty_gpu():
    """Returns the first empty GPU on the machine as gpu<id> for passing to nmtpy."""
    root = ET.fromstring(subprocess.check_output(['nvidia-smi', '-x', '-q']))
    for idx, gpu in enumerate(root.findall('gpu')):
        if gpu.find('processes').find('process_info') is None:
            return 'gpu%d' % idx
    return None

def reap_processes():
    """Remove a process and decrease semaphore if worker is finished."""
    for k,v in PROCS.items():
        # Check alive
        if v.poll() is not None:
            del PROCS[k]
            SEMAPHORE_GPU.release()
    # Return number of free GPUs
    return N_GPU - len(PROCS)

def sample_dropout(vmax=8):
    """Sample dropout probability <= vmax."""

    return {
            'emb_dropout' : (np.random.randint(1, vmax) / 10.),
            'ctx_dropout' : (np.random.randint(1, vmax) / 10.),
            'out_dropout' : (np.random.randint(1, vmax) / 10.),
           }

def generate_experiments():
    """Create parameter sets for each experiment."""
    for seed in SEEDS:
        for dim in DIMS:
            for arch in ARCHS:
                EXPERIMENTS.append({
                    'model-type'    : arch,
                    'seed'          : seed,
                    'embedding-dim' : dim,
                    'rnn-dim'       : dim,
                    })

def spawn_trainer(conf, params):
    """Spawn GPU trainer and return the process instance."""
    gpuid = get_empty_gpu()
    if gpuid:
        cmd = ['nmt-train', '-D', gpuid, '-c', conf, '-R', str(params['seed']),
                            '-T', params['model-type'], '-e']
        del params['seed']
        del params['model-type']

        # Push parameters
        for key, value in params.iteritems():
            cmd.append('%s:%s' % (key, value))

        # Set PYTHONUNBUFFERED for being able to capture std streams
        env = os.environ
        env['PYTHONUNBUFFERED'] = 'YES'

        # Generate a UUID for each system
        return (uuid.uuid4(), subprocess.Popen(cmd, stdout=DEVNULL, stderr=DEVNULL, env=env))
    else:
        return None, None

def spawn_single(config, expdict):
    """Spawns a single instance."""
    if not SEMAPHORE_GPU.acquire(blocking=False):
        return None, None

    uid, ps = spawn_trainer(config, expdict)

    # No more GPUs available, decrease semaphore
    if uid is None:
        SEMAPHORE_GPU.release()
        return None, None

    return uid, ps

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='modelsel')
    parser.add_argument('-c', '--config'    , type=str, help='Base configuration file.', required=True)
    parser.add_argument('-m', '--max-epochs', type=int, help='Max epoch for each run.', default=30)
    parser.add_argument('-s', '--diffseed'  , help='Launch each system with 3 deterministic seeds', action='store_true', default=False)

    args = parser.parse_args()

    # Keep 3 different seeds?
    if args.diffseed:
        SEEDS.extend([1235, 1236])

    # Get # of GPUs and create a semaphore
    N_GPU = get_gpu_count()
    print '# of GPUs available: %d' % N_GPU
    SEMAPHORE_GPU = Semaphore(N_GPU)

    default_params = {'max-epochs'  : args.max_epochs,
                      'beam-size'   : 3,
                     }

    generate_experiments()
    n_experiments = len(EXPERIMENTS)
    print '# of total experiments: %d' % n_experiments

    while len(EXPERIMENTS) > 0:
        # Cleanup finished workers if any
        n_free = reap_processes()
        for i in range(min(n_free, len(EXPERIMENTS))):
            # Fetch next experiment parameters
            expdict = EXPERIMENTS[0]
            # Add default params
            expdict.update(default_params)

            uid, ps = spawn_single(args.config, dict(expdict))
            if uid is not None:
                PROCS[uid] = ps
                # Remove the experiment
                print '[PID=%7d] %30s %s' % (ps.pid, uid, expdict)
                EXPERIMENTS.pop(0)
                time.sleep(5)
            else:
                print 'spawn_single() failed.'

            print '(%d experiments left to launch)' % len(EXPERIMENTS)

        # Sleep for 3 minutes
        time.sleep(60*3)

    # Wait for final experiments
    while reap_processes() != N_GPU:
        time.sleep(60*3)