# -*- coding: utf-8 -*-
import os
import bz2
import sys
import copy
import gzip
import lzma
import tempfile
import subprocess

from . import cleanup

def print_summary(train_args, model_args, print_func=None):
    """Returns or prints a summary of training/model options."""
    def _get_max_width(keys):
        return max([len(k) for k in keys]) + 1

    def _dict_str(d, maxlen):
        res = ""
        templ = '%' + str(maxlen) + 's : '
        kvs = []
        for k,v in list(d.items()):
            if isinstance(v, list):
                kvs.append((k, v.pop(0)))
                for l in v:
                    kvs.append((k, l))
            else:
                kvs.append((k,v))

        kvs = sorted(kvs, key=lambda x: x[0])
        for k,v in kvs:
            res += (templ % k) + str(v) + '\n'
        return res

    train_args = copy.deepcopy(train_args)
    model_args = copy.deepcopy(model_args)
    max_width = _get_max_width(list(train_args.__dict__.keys()) +
                               list(model_args.__dict__.keys()))

    # Add training options
    result  = 'Training options:'
    result += '\n' + ('-' * 35) + '\n'

    result += _dict_str(train_args.__dict__, max_width)

    # Copy
    model_args = dict(model_args.__dict__)
    # Remove these and treat them separately
    model_data = model_args.pop('data')
    model_dict = model_args.pop('dicts')

    # Add model options
    result += '\nModel options:'
    result += '\n' + ('-' * 35) + '\n'

    result += _dict_str(model_args, max_width)
    result += ('%' + str(max_width) + 's =\n') % 'dicts'
    result += _dict_str(model_dict, max_width)
    result += ('%' + str(max_width) + 's =\n') % 'data'
    result += _dict_str(model_data, max_width)

    if print_func:
        for line in result.split('\n'):
            print_func(line)
    else:
        return result

def pretty_dict(elem, msg=None, print_func=None):
    """Returns a string representing elem optionally prepended by a message."""
    result = ""
    if msg:
        # Add message
        result += msg + '\n'
        # Add trailing lines
        result += ('-' * len(msg)) + '\n'

    skeys = sorted(elem.keys())
    maxlen = max([len(k) for k in skeys]) + 1
    templ = '%' + str(maxlen) + 's : '
    for k in skeys:
        result += (templ % k) + str(elem[k]) + '\n'

    if print_func:
        for line in result.split('\n'):
            print_func(line)
    else:
        return result

def ensure_dirs(dirs):
    """Create a list of directories if not exists."""
    try:
        for d in dirs:
            os.makedirs(d)
    except OSError as oe:
        pass

def real_path(p):
    """Expand UNIX tilde and return real path."""
    return os.path.realpath(os.path.expanduser(p))

def force_symlink(origfile, linkname, relative=False):
    if relative:
        origfile = os.path.basename(origfile)
    try:
        os.symlink(origfile, linkname)
    except FileExistsError as e:
        os.unlink(linkname)
        os.symlink(origfile, linkname)

def listify(l):
    """Encapsulate l with list[] if not."""
    return [l] if not isinstance(l, list) else l

def readable_size(n):
    """Return a readable size string."""
    sizes = ['K', 'M', 'G']
    fmt = ''
    size = n
    for i,s in enumerate(sizes):
        nn = n / (1000.**(i+1))
        if nn >= 1:
            size = nn
            fmt = sizes[i]
        else:
            break
    return '%.1f%s' % (size, fmt)

def get_temp_file(suffix="", name=None, delete=False):
    """Creates a temporary file under /tmp."""
    if name:
        name = os.path.join("/tmp", name)
        t = open(name, "w")
        cleanup.register_tmp_file(name)
    else:
        _suffix = "_nmtpy_%d" % os.getpid()
        if suffix != "":
            _suffix += suffix

        t = tempfile.NamedTemporaryFile(suffix=_suffix, delete=delete)
        cleanup.register_tmp_file(t.name)
    return t

def get_valid_evaluation(save_path, beam_size, n_jobs, mode, metric,
                         valid_mode='single', trans_cmd='nmt-translate', f_valid_out=None, factors=None):
    """Run nmt-translate for validation during training."""
    cmd = [trans_cmd, "-b", str(beam_size), "-D", mode,
           "-j", str(n_jobs), "-m", save_path, "-M", metric, "-v", valid_mode]
    # Factors option needs -fa option with the script and 2 output files
    if factors:
        cmd.extend(["-fa", factors, "-o", f_valid_out[0], f_valid_out[1]])

    elif f_valid_out is not None:
        cmd.extend(["-o", f_valid_out])

    # nmt-translate will print a dict of metrics
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=sys.stdout)
    cleanup.register_proc(p.pid)
    out, err = p.communicate()
    cleanup.unregister_proc(p.pid)

    # Return None if nmt-translate failed
    if p.returncode != 0:
        return None

    out = out.splitlines()[-1]
    # Convert metrics back to dict
    return eval(out.strip())

def create_gpu_lock(used_gpu):
    """Create a lock file for GPU reservation."""
    name = "gpu_lock.pid%d.gpu%s" % (os.getpid(), used_gpu)
    lockfile = get_temp_file(name=name)
    lockfile.write("[nmtpy] %s\n" % name)

def fopen(filename, mode=None):
    """GZ/BZ2/XZ-aware file opening function."""
    # NOTE: Mode is not used but kept for not breaking iterators.
    if filename.endswith('.gz'):
        return gzip.open(filename, 'rt')
    elif filename.endswith('.bz2'):
        return bz2.open(filename, 'rt')
    elif filename.endswith(('.xz', '.lzma')):
        return lzma.open(filename, 'rt')
    else:
        # Plain text
        return open(filename, 'r')

def find_executable(fname):
    """Find executable in PATH."""
    fname = os.path.expanduser(fname)
    if os.path.isabs(fname) and os.access(fname, os.X_OK):
        return fname
    for path in os.environ['PATH'].split(':'):
        fpath = os.path.join(path, fname)
        if os.access(fpath, os.X_OK):
            return fpath

def get_device(which='auto'):
    """Return Theano device to use by favoring GPUs first."""
    # Use CPU
    if which == "cpu":
        return "cpu", None

    # Use the requested GPU without looking for availability
    elif which.startswith("gpu"):
        create_gpu_lock(int(which.replace("gpu", "")))
        return which

    # auto favors GPU in the first place
    elif which == 'auto':
        try:
            p = subprocess.run(["nvidia-smi", "-q", "-d", "PIDS"], stdout=subprocess.PIPE, universal_newlines=True)
        except FileNotFoundError as oe:
            # Binary not found, fallback to CPU
            return "cpu"

        # Find out about GPU usage
        usage = ["None" in l for l in p.stdout.split("\n") if "Processes" in l]

        try:
            # Get first unused one
            which = usage.index(True)
        except ValueError as ve:
            # No available GPU on this machine
            return "cpu"

        create_gpu_lock(which)
        return ("gpu%d" % which)

def get_exp_identifier(train_args, model_args, suffix=None):
    """Return a representative string for the experiment."""

    names = [train_args.model_type]

    for k in sorted(model_args.__dict__):
        if k.endswith("_dim"):
            # Only the first letter should suffice for now, e for emb, r for rnn
            names.append('%s%d' % (k[0], getattr(model_args, k)))

    # Join so far
    name = '-'.join(names)

    # Append optimizer and learning rate
    name += '-%s_%.e' % (model_args.optimizer, model_args.lrate)

    # Append batch size
    name += '-bs%d' % model_args.batch_size

    # Validation stuff (first: early-stop metric)
    name += '-%s' % train_args.valid_metric.split(',')[0]

    if train_args.valid_freq > 0:
        name += "-each%d" % train_args.valid_freq
    else:
        name += "-eachepoch"

    if train_args.decay_c > 0:
        name += "-l2_%.e" % train_args.decay_c

    # Dropout parameter names can be different for each model
    dropouts = sorted([opt for opt in model_args.__dict__ \
                           if opt.endswith('dropout')])
    if len(dropouts) > 0:
        name += "_do"
        for dout in dropouts:
            name += "_%s%.1f" % (dout[0], model_args.__dict__[dout])

    if train_args.clip_c > 0:
        name += "-gc%d" % int(train_args.clip_c)

    if isinstance(model_args.weight_init, str):
        name += "-init_%s" % model_args.weight_init
    else:
        name += "-init_%.2f" % model_args.weight_init

    # Append seed
    name += "-s%d" % train_args.seed

    if suffix:
        name = "%s-%s" % (name, suffix)

    return name

def get_next_runid(save_path, exp_id):
    # Log file, runs start from 1, incremented if exists
    i = 1
    while os.path.exists(os.path.join(save_path, "%s.%d.log" % (exp_id, i))):
        i += 1

    return i
