![nmtpy](docs/logo.png?raw=true "nmtpy")

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**nmtpy** is a suite of Python tools, primarily based on the starter code provided in [dl4mt-tutorial](https://github.com/nyu-dl/dl4mt-tutorial) for training neural machine translation networks using Theano.

The basic motivation behind forking **dl4mt-tutorial** was to create a framework where it would be
easy to implement a new model by just copying and modifying an existing model class (or even
inheriting from it and overriding some of its methods).

To achieve this purpose, **nmtpy** tries to completely isolate training loop, beam search,
iteration and model definition:
  - `nmt-train` script to initiate a training experiment
  - `nmt-translate` to produce model-agnostic translations. You just pass a trained model's
  checkpoint file and it does its job.
  - An abstract `BaseModel` class to derive from to define your NMT architecture.
  - An abstract `Iterator` to derive from for your custom iterators.

A non-exhaustive list of differences between **nmtpy** and **dl4mt-tutorial** is as follows:

#### General/API
  - No shell script, everything is in Python 
  - Overhaul object-oriented refactoring of the code: clear separation of API and scripts that interface with the API
  - INI style configuration files to define everything regarding a training experiment
  - Transparent cleanup mechanism to kill stale processes, remove temporary files
  - Simultaneous logging of training details to stdout and log file
  
#### Training/Inference
  - Supports out-of-the-box BLEU, METEOR and COCO eval metrics
  - Includes [subword-nmt](https://github.com/rsennrich/subword-nmt) utilities for training and applying BPE model
  - Plugin-like text filters for hypothesis post-processing (Example: BPE, Compound)
  - Early-stopping and checkpointing based on perplexity, BLEU or METEOR
    - `nmt-train` automatically calls `nmt-translate` during validation and returns the result back
    - Ability to add new metrics easily
  - Single `.npz` file to store everything about a training experiment
  - Automatic free GPU selection and reservation using `nvidia-smi`
  - Shuffling support between epochs:
    - Simple shuffle
    - [Homogeneous batches of same-length samples](https://github.com/kelvinxu/arctic-captions) to improve training speed
  - Improved parallel translation decoding on CPU
  - Forced decoding i.e. rescoring using NMT
  - Export decoding informations into `json` for further visualization of attention coefficients
  
#### Deep Learning
  - Improved numerical stability and reproducibility
  - Glorot/Xavier, He, Orthogonal weight initializations
  - Efficient SGD, Adadelta, RMSProp and ADAM
    - Single forward/backward theano function without intermediate variables
  - Ability to stop updating a set of weights by recompiling optimizer
  - Several recurrent blocks:
    - GRU, Conditional GRU (CGRU) and LSTM
    - Multimodal attentive CGRU variants
  - [Layer Normalization](https://github.com/ryankiros/layer-norm) support for GRU
  - [Tied target embeddings](https://arxiv.org/abs/1608.05859)
  - Simple/Non-recurrent Dropout, L2 weight decay
  - Training and validation loss normalization for comparable perplexities
  - Initialization of a model with a pretrained NMT for further finetuning

## Models

### Attentional NMT: `attention.py`
This is the basic shallow attention based NMT from `dl4mt-tutorial` improved in different ways:
  - 3 forward dropout layers after source embeddings, source context and before softmax managed by the configuration parameters `emb_dropout, ctx_dropout, out_dropout`.
  - Layer normalization for source encoder (`layer_norm=True|False`)
  - Tied target embeddings (`tied_trg_emb=True|False`)
  
This model uses the simple `BitextIterator` i.e. it directly reads plain parallel text files as defined in the experiment configuration file. Please see [this monomodal example](examples/wmt16-mmt-task1/wmt16-mmt-task1-monomodal.conf) for usage.

### Multimodal NMT / Image Captioning: `fusion*py`

These `fusion` models derived from `attention.py` and `basefusion.py` implement several multimodal NMT / Image Captioning architectures detailed in the following papers:

[Caglayan, Ozan, et al. "Does Multimodality Help Human and Machine for Translation and Image Captioning?." arXiv preprint arXiv:1605.09186 (2016).](https://arxiv.org/abs/1605.09186)

[Caglayan, Ozan, Loïc Barrault, and Fethi Bougares. "Multimodal Attention for Neural Machine Translation." arXiv preprint arXiv:1609.03976 (2016).](https://arxiv.org/abs/1609.03976)

The models are separated into 8 files implementing their own multimodal CGRU differing in the way the attention is formulated in the decoder (4 ways) x the way the multimodal contexts are fusioned (2 ways: SUM/CONCAT). These models also use a different data iterator, namely `WMTIterator` that requires converting the textual data into `.pkl` as in the [multimodal example](examples/wmt16-mmt-task1).

The `WMTIterator` only knows how to handle the ResNet-50 convolutional features that we provide in the examples page. If you would like to use FC-style fixed-length vectors or other types of multimodal features, you need to write your own iterator.

### Factored NMT: `attention_factors.py`

The model file `attention_factors.py` corresponds to the following paper:

[García-Martínez, Mercedes, Loïc Barrault, and Fethi Bougares. "Factored Neural Machine Translation." arXiv preprint arXiv:1609.04621 (2016).](https://arxiv.org/abs/1609.04621)

In the examples folder of this repository, you can find data and a configuration file to run this model.

### RNNLM: `rnnlm.py`

This is a basic recurrent language model to be used with `nmt-test-lm` utility.





## Requirements

You need the following Python libraries installed in order to use **nmtpy**:
  - numpy
  - Theano >= 0.8 (0.9.x would be better)

- We recommend using Anaconda Python distribution which is equipped with Intel MKL (Math Kernel Library) greatly
  improving CPU decoding speeds during beam search. With a correct compilation and installation, you should achieve
  similar performance with OpenBLAS as well but the setup procedure may be difficult to follow for inexperienced ones.
- nmtpy only supports Python 3, please see [pythonclock.org](http://pythonclock.org)
- Please note that METEOR requires a **Java** runtime so `java` should be in your `$PATH`.

#### Additional data for METEOR

Before installing **nmtpy**, you need to run `scripts/get-meteor-data.sh` to download METEOR paraphrase files.

#### Installation

```
$ python setup.py install
```

**Note:** When you add a new model under `models/` it will not be directly available in runtime
as it needs to be installed as well. To avoid re-installing each time, you can use development mode with `python setup.py develop` which will directly make Python see the `git` folder as the library content.

## Ensuring Reproducibility in Theano

When we started to work on **dl4mt-tutorial**, we noticed an annoying reproducibility problem where
multiple runs of the same experiment (same seed, same machine, same GPU) were not producing exactly
the same training and validation losses after a few iterations.

The first solution that was [discussed](https://github.com/Theano/Theano/issues/3029) in Theano
issues was to replace a non-deterministic GPU operation with its deterministic equivalent. To achieve this,
you should **patch** your local Theano installation using [this patch](patches/00-theano-advancedinctensor.patch) (or [this one](patches/00-theanov0.9-advancedinctensor.patch) for the recent master which approaches v0.9) unless upstream developers add a configuration option to `.theanorc`.

But apparently this was not enough to obtain reproducible models. After debugging ~2 months, we discovered and
[fixed](https://github.com/Theano/Theano/commit/8769382ff661aab15dda474a4c74456037f73cc6) a very insidious bug involving back-propagation in Theano.

So if you care (and you absolutely should) about reproducibility, make sure your Theano copy has above changes applied. If your Theano copy is newer than *17 August 2016*, the second fix should be available in your copy.

## Configuring Theano

Here is a basic `.theanorc` file (Note that the way you install CUDA, CuDNN
may require some modifications):

```
[global]
# Not so important as nmtpy will pick an available GPU
device = gpu0
# We use float32 everywhere
floatX = float32
# Keep theano compilation in RAM if you have a 7/24 available server
base_compiledir=/tmp/theano-%(user)s

[cuda]
# CUDA 8.0 is better
root = /opt/cuda-7.5

[dnn]
# Make sure you use CuDNN as well
enabled = auto
library_path = /opt/CUDNN/cudnn-v5.1/lib64
include_path = /opt/CUDNN/cudnn-v5.1/include

[lib]
# Allocate 95% of GPU memory once
cnmem = 0.95
```

If you have a recent Theano, you may want to try the new GPU backend after
installing [libgpuarray](https://github.com/Theano/libgpuarray). In order to do so,
pass `GPUARRAY=1` into the environment when running `nmt-train`:

```
$ GPUARRAY=1 nmt-train -c <conf file> ...
```

Note that we could not obtain accurate results using Maxwell GPUs with this backend
so use it at your own risk.

### Checking BLAS configuration

Recent Theano versions can automatically detect correct MKL flags. You should obtain a similar output after running the following command:

```
$ python -c 'import theano; print theano.config.blas.ldflags'
-L/home/ozancag/miniconda/lib -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -lm -Wl,-rpath,/home/ozancag/miniconda/lib
```

## Acknowledgements

**nmtpy** includes code from the following projects:

 - [dl4mt-tutorial](https://github.com/nyu-dl/dl4mt-tutorial)
 - Scripts from [subword-nmt](https://github.com/rsennrich/subword-nmt)
 - Ensembling and alignment collection from [nematus](https://github.com/rsennrich/nematus)
 - `multi-bleu.perl` from [mosesdecoder](https://github.com/moses-smt/mosesdecoder)
 - METEOR v1.5 JAR from [meteor](https://github.com/cmu-mtlab/meteor)
 - Sorted data iterator, coco eval script and LSTM from [arctic-captions](https://github.com/kelvinxu/arctic-captions)
 - `pycocoevalcap` from [coco-caption](https://github.com/tylin/coco-caption)

See [LICENSE](LICENSE.md) file for license information.
