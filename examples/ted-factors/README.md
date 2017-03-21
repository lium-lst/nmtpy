# Factored Neural Machine Translation system

This system generates multiple outputs for the neural network.

## TED data 

- Train and dev data are stored under the folder `data/`

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
nmt-translate -m ~/nmtpy/models/<your model file> \
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


