# Factored Neural Machine Translation system

The Factored NMT models defined by ```basefnmt.py``` are based on the NMT architecture and extended to be able to generate several output symbols at the same time (Figure http://www-lium.univ-lemans.fr/~garcia/fnmt_archi.pdf).

The decoder has been modified respect to the baseline model with the following items:

- Specialized iterator named ```factors.py``` that handles multiple inputs and outputs text streams.
- Additional softmax and embedding for the 2nd output.
- Concatenation of the embeddings of the generated tokens at previous timestep to feedback the generation of the current token.
- Sum of costs coming from each output.
- Constriction of the length of the 2nd output sequence to be equal to the length of the 1st output sequence. 
Firstly, we included a new mask excluding the end of sequence (\tm{EOS}) symbols to avoid shorter sequences. 
Secondly, we limited the maximum length of the 2nd output sequence to the length of the 1st output sequence.
- The beam search has been modified to be able to handle the multiple outputs.
Once we obtain the hypothesis from lemmas (1st output) and factors (2nd output) at stage 1 of the Figure http://www-lium.univ-lemans.fr/~garcia/beamsearch.pdf, the cross product of those output spaces is performed.
Afterwards, we keep the beam size best combinations for each hypothesis. 
Finally, the number of samples is reduced again to the beam size.
- Translation generation executed by ```nmt-translate-factors``` which can handle multiple outputs. 
- Optionally, \tm{factors2wordbleu.py} metric is available to evaluate with BLEU the combination of the several outputs. 
A script detailed in the configuration file is necessary to apply this metric.

## TED data 

- Download [examples-ted-data.tar.bz2](http://www-lium.univ-lemans.fr/~garcia/examples-ted-data.tar.bz2) and extract it into the `data/` folder.

- Build the vocabulary dictionaries for each train file:

`nmt-build-dict train_file`

- Option factors enable the factored system.
Factors parameter gets as argument `evalf` which will evaluate the model just with the first output or a script to combine the 2 outputs as desired.

This script will need as arguments `lang, first_output_hyp_file, second_output_hyp_file, reference_file` in this order and will print the corresponding BLEU score.

## FNMT Training

Run `nmt-train -c attention_factors-ted-en-fr.conf` to train a FNMT on this corpus. 

## FNMT Translation

When the training is over, you can translate the test set using the following command:

```
nmt-translate-factors -m ~/nmtpy/models/<your model file> \
                      -S ~/nmtpy/examples/ted-factors/data/dev.en \
                      -R ~/nmtpy/examples/ted-factors/data/dev.fr \
                         ~/nmtpy/examples/ted-factors/data/dev.lemma.fr \
                         ~/nmtpy/examples/ted-factors/data/dev.factors.fr \
                      -o trans_dev.lemma.fr trans_dev.factors.fr \
                      -fa evalf
```
The option -R needs the references of the word-level, first output and second output, repectively.

In -fa option you can include your script to combine both outputs if desired instead of evalf option.


## Citation:
If you use `fnmt` system in your work, please cite the following:

```
@inproceedings{garcia-martinez2016fnmt,
  title={Factored Neural Machine Translation Architectures},
  author={Garc{\'\i}a-Mart{\'\i}nez, Mercedes and Barrault, Lo{\"\i}c and Bougares, Fethi},
  booktitle={arXiv preprint arXiv:1605.09186},
  year={2016}
}
```

More info:
http://workshop2016.iwslt.org/downloads/IWSLT_2016_paper_2.pdf

Contact: Mercedes.Garcia_Martinez@univ-lemans.fr.


