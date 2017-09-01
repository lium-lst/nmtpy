#!/bin/bash

# Example script to show how to prepare
# a char2char dataset from raw text corpora.
# You need to use filter:char2words in .conf
# to correctly post-process hypotheses after beam-search

# Pipeline:
#  lowercase.perl -> word2char (sed) -> trim whitespace (awk)

datadir=../
SL=en
TL=de

for dataset in train val test2016 test2017; do
  for lang in $SL $TL; do
    inputfile=${datadir}/${dataset}.${lang}
    if [ -f $inputfile ]; then
      echo $dataset, $lang
      lowercase.perl -l $SL < $inputfile | sed -e "s/./& /g;s/\ \ \ / <s> /g" \
        | awk '{$1=$1};1' > ${dataset}.lc.char.${lang}
    fi
  done
done

nmt-build-dict train*
