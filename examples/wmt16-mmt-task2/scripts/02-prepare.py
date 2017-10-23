#!/usr/bin/env python
import os
import string
import pickle
import argparse

from collections import OrderedDict, Counter

import numpy as np
# Each element of the list that is pickled is in the following format:
# [ssplit, tsplit, imgid, imgname, swords, twords]

puncs = list(string.punctuation)
puncs += ["..", "...", "&amp;", "&apos;", "&gt;", "&lt;", "&quot;"]

_ = os.path.expanduser

def create_dictionaries(sents, outpath, minvocab):
    words = [[], []]
    names = ["train_src.pkl", "train_trg.pkl"]

    for s in sents:
        words[0].extend(s[4])
        words[1].extend(s[5])

    for i in [0, 1]:
        freqs = Counter(words[i]).most_common()
        vocab = OrderedDict()
        with open(os.path.join(outpath, names[i]), 'wb') as df:
            vocab['<eos>'] = 0
            vocab['<unk>'] = 1

            tokens  = [w for w,f in freqs]
            freqs  = [f for w,f in freqs]

            sorted_idx = np.argsort(freqs)
            sorted_words = [tokens[ii] for ii in sorted_idx[::-1] if freqs[ii] >= minvocab]

            for ii, ww in enumerate(sorted_words):
                vocab[ww] = ii + 2

            print("%s: %d words" % (names[i], len(vocab)))

            pickle.dump(vocab, df)

def process_file(fname, lowercase, strippunc):
    caps = open(fname)
    result = []
    for line in caps:
        line = line.strip()
        if lowercase:
            line = line.lower()

        # Split sentences
        words = line.split(" ")

        if strippunc:
            words = [w for w in words if w not in puncs]

        result.append(words)

    caps.close()
    return result

def process_sentence_pairs(imgs, src_sents, trg_sents, sidx, tidx, clean=True,
                           minlen=3, maxlen=50, ratio=3, img_offset=0):
    pairs = []

    for idx, (swords, twords) in enumerate(zip(src_sents, trg_sents)):
        slen = len(swords)
        tlen = len(twords)

        if clean:
            len_ratio = float(slen) / float(tlen)

            # This is like clean-corpus-frac-n.perl script
            if slen < minlen or tlen < minlen or \
               slen > maxlen or tlen > maxlen or \
               len_ratio > ratio or (1./len_ratio) > ratio:
                continue

        pairs.append([sidx, tidx, idx, imgs[img_offset+idx], swords, twords])

    return pairs

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='prepare')
    parser.add_argument('-l', '--lowercase' , help='Do lowercasing', action='store_true')
    parser.add_argument('-s', '--strippunc' , help='Strip punctuation', action='store_true')
    parser.add_argument('-m', '--minlen'    , help='Minimum sentence length', default=3, type=int)
    parser.add_argument('-M', '--maxlen'    , help='Maximum sentence length', default=50, type=int)
    parser.add_argument('-r', '--ratio'     , help='src length to trg length min ratio', default=3, type=int)
    parser.add_argument('-d', '--minvocab'  , help='Consider words occuring at least this number', default=1, type=int)
    parser.add_argument('-o', '--output'    , help='Output directory.', default="./processed")

    parser.add_argument('-i', '--imagelist'  , help='Image list file.', required=True)

    parser.add_argument('-t', '--trainsrc'   , nargs='+', help="Training source sentence file(s).", required=True)
    parser.add_argument('-T', '--traintrg'   , nargs='+', help="Training target sentence file(s).", required=True)
    parser.add_argument('-v', '--validsrc'   , nargs='+', help="Validation source sentence file(s).", required=True)
    parser.add_argument('-V', '--validtrg'   , nargs='+', help="Validation target sentence file(s).", required=True)
    parser.add_argument('-e', '--testsrc'    , nargs='+', help="Test source sentence file(s).", required=False)
    parser.add_argument('-E', '--testtrg'    , nargs='+', help="Test target sentence file(s).", required=False)

    args = parser.parse_args()

    # Load image list. This contains the order of the images
    # in the features file.
    imgs = open(args.imagelist).read().strip().split("\n")
    print("# of images in the image list: %d" % len(imgs))

    # Create output directory if not available
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # Sort files so that they're in order
    train_src_files = sorted(args.trainsrc)
    train_trg_files = sorted(args.traintrg)
    valid_src_files = sorted(args.validsrc)
    valid_trg_files = sorted(args.validtrg)
    test_src_files  = []
    if args.testsrc:
        test_src_files = sorted(args.testsrc)
        test_trg_files = sorted(args.testtrg)

    srclang = train_src_files[0].split(".")[-1]
    trglang = train_trg_files[0].split(".")[-1]

    # Print some informations
    print("Lowercase: ", args.lowercase)
    print("Strip punctuations: ", args.strippunc)
    print("Min sentence length: %d" % args.minlen)
    print("Max sentence length: %d" % args.maxlen)
    print("Min word freq for vocab: %d" % args.minvocab)
    print("Found %d training files." % len(train_src_files))
    print("Found %d validation files." % len(valid_src_files))
    if len(test_src_files) > 0:
        print("Found %d test files." % len(test_src_files))
    print("Source language: %s" % srclang)
    print("Target language: %s" % trglang)

    assert len(train_src_files) == len(train_trg_files)
    assert len(valid_src_files) == len(valid_trg_files)

    # Number of input sentence files
    n_splits = len(train_src_files)

    train_src_sents = []
    train_trg_sents = []
    valid_src_sents = []
    valid_trg_sents = []
    test_src_sents  = []
    test_trg_sents  = []

    # Read all sentences once, do lowercasing and strip punctuations if requested
    for f in train_src_files:
        print(f)
        train_src_sents.append(process_file(f, args.lowercase, args.strippunc))

    for f in train_trg_files:
        print(f)
        train_trg_sents.append(process_file(f, args.lowercase, args.strippunc))

    for f in valid_src_files:
        print(f)
        valid_src_sents.append(process_file(f, args.lowercase, args.strippunc))

    for f in valid_trg_files:
        print(f)
        valid_trg_sents.append(process_file(f, args.lowercase, args.strippunc))

    for f in test_src_files:
        print(f)
        test_src_sents.append(process_file(f, args.lowercase, args.strippunc))

    for f in test_trg_files:
        print(f)
        test_trg_sents.append(process_file(f, args.lowercase, args.strippunc))

    sent_sets = {'train' : (train_src_sents, train_trg_sents),
                 'valid' : (valid_src_sents, valid_trg_sents),
                 'test'  : (test_src_sents, test_trg_sents)
                }

    if args.lowercase or args.strippunc:
        print("Rewriting processed validation sentences")
        # Rewrite validation files as they are now processed
        suffix = ""
        if args.lowercase:
            suffix += ".lc"
        if args.strippunc:
            suffix += ".nopunct"

        for s in ['valid', 'test']:
            src_sents, trg_sents = sent_sets[s]
            for idx, split in enumerate(src_sents, 1):
                with open(os.path.join(args.output, '%s.%d.tok%s.%s' % (s, idx, suffix, srclang)), 'w') as f:
                    for sent in split:
                        f.write(" ".join(sent) + '\n')

            for idx, split in enumerate(trg_sents, 1):
                with open(os.path.join(args.output, '%s.%d.tok%s.%s' % (s, idx, suffix, trglang)), 'w') as f:
                    for sent in split:
                        f.write(" ".join(sent) + '\n')

    tr_sum = sum([len(spl) for spl in train_src_sents])
    vl_sum = sum([len(spl) for spl in valid_src_sents])
    ts_sum = sum([len(spl) for spl in test_src_sents])
    print("Total number of training sentences: %d" % (tr_sum))
    print("Total number of validation sentences: %d" % (vl_sum))
    print("Total number of test sentences: %d" % (ts_sum))

    # Specific to Flickr30k
    n_train = 29000
    n_valid = 1014
    n_test  = 1000

    sentences = {'train' : [], 'valid' : [], 'test' : []}
    splits = {
                'train' : [train_src_sents, train_trg_sents, 0],
                'valid' : [valid_src_sents, valid_trg_sents, n_train],
                'test'  : [test_src_sents,  test_trg_sents, n_train + n_valid],
             }

    for split in ['train', 'valid', 'test']:
        do_clean = (split == 'train')
        print("Processing %s splits" % split)
        # This is the cross product of all splits
        src_sents, trg_sents, img_offset = splits[split]
        for sidx, ssentsplit in enumerate(src_sents):
            print("%d => " % len(ssentsplit), end=' ') 
            for tidx, tsentsplit in enumerate(trg_sents):
                print(len(tsentsplit), end=' ') 
                sentences[split].extend(process_sentence_pairs(imgs, ssentsplit, tsentsplit, sidx, tidx,
                                        clean=do_clean, minlen=args.minlen, maxlen=args.maxlen, ratio=args.ratio, img_offset=img_offset))
            print()

        print("Final %s pair count: %d" % (split, len(sentences[split])))

    for split in ['train', 'valid', 'test']:
        if len(sentences[split]) > 0:
            print("Dumping %s pkl file" % split)
            with open(os.path.join(args.output, 'flickr_30k_align.%s.pkl' % split), 'wb') as f:
                pickle.dump(sentences[split], f)

    # Create the dictionary with nmt-build-dict afterwards
    create_dictionaries(sentences['train'], args.output, args.minvocab)
