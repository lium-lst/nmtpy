#!/usr/bin/env python
import os
import sys
import pickle
import argparse

from itertools import zip_longest

# Each element of the list that is pickled is in the following format:
# [ssplit, tsplit, imgid, imgname, swords, twords]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='prepare')

    parser.add_argument('-o', '--outfile', help='Output file name',     required=True, type=argparse.FileType('wb'))
    parser.add_argument('-i', '--imglist', help='Image list file',      required=True, type=argparse.FileType('r'))
    parser.add_argument('-s', '--source', help="Source sentence file",  required=True, type=argparse.FileType('r'))
    parser.add_argument('-t', '--target', help="Target sentence file",  default=[],    type=argparse.FileType('r'))
    parser.add_argument('-l', '--lines', help="Lines retained file",    default=None,  type=str)

    args = parser.parse_args()

    imglist = args.imglist.read().strip().split('\n')

    print('# of images: {0}'.format(len(imglist)))

    # By default each sentence maps to its relevant line in the
    # imglist file (1-1 mapping)
    lines = list(range(len(imglist)))
    if args.lines:
        # Image-sentence pair mapping should be done
        # based on the 'lines retained' file (NOTE: 1-indexed file!)
        with open(args.lines) as f:
            lines = [int(s)-1 for s in f.read().strip().split('\n')]

    # This should be == lines in the train files
    print('# of lines retained: {0}'.format(len(lines)))

    seqs = []

    # Read sentence pairs
    for idx, (ssent, tsent) in enumerate(zip_longest(args.source, args.target)):
        pair_id = lines[idx]
        seqs.append([None, None, pair_id, imglist[pair_id], ssent.strip().split(' '), tsent.strip().split(' ') if tsent else None])

    pickle.dump(seqs, args.outfile)
