# Examples

In this folder you can find ready-to-train baseline models for several tasks/architectures.

## WMT16 Shared Task on Multimodal Translation

 - [Multimodal Translation (Task 1)](wmt16-mmt-task1) : Monomodal and Multimodal
 - [Cross-lingual Image Description Generation (Task 2)](wmt16-mmt-task2) : Monomodal and Multimodal

**Note:** All textual data provided in the `data/` folders of the above examples are the courtesy of the following work
and can be downloaded from [here](http://www.statmt.org/wmt16/multimodal-task.html):

```
@article{elliott-EtAl:2016:VL16,
 author    = {{Elliott}, D. and {Frank}, S. and {Sima'an}, K. and {Specia}, L.},
 title     = {Multi30K: Multilingual English-German Image Descriptions},
 booktitle = {Proceedings of the 5th Workshop on Vision and Language},
 year      = {2016},
 pages     = {70--74},
 year      = 2016
}
```

If you use `fusion_*` multimodal architectures in your work, please cite the following
article:

```
@article{caglayan2016multimodal,
  title={Multimodal Attention for Neural Machine Translation},
  author={Caglayan, Ozan and Barrault, Lo{\"\i}c and Bougares, Fethi},
  journal={arXiv preprint arXiv:1609.03976},
  year={2016}
}
```

### Getting the Image Features

For multimodal baselines, you will need the convolutional features extracted
from a pre-trained ResNet-50. You can download these files from the links below:

 - [flickr30k_ResNets50_blck4_train.fp16.npy.xz](http://www-lium.univ-lemans.fr/sites/default/files/NMTPY/flickr30k_ResNets50_blck4_train.fp16.npy.xz) (6GB)
 - [flickr30k_ResNets50_blck4_val.fp16.npy.xz](http://www-lium.univ-lemans.fr/sites/default/files/NMTPY/flickr30k_ResNets50_blck4_val.fp16.npy.xz) (214M)
 - [flickr30k_ResNets50_blck4_test.fp16.npy.xz](http://www-lium.univ-lemans.fr/sites/default/files/NMTPY/flickr30k_ResNets50_blck4_test.fp16.npy.xz) (211M)

After downloading the files, extract them using the following command:

```
xz -d <downloaded xz file>
```

Each `.npy` file contains 14x14x1024 convolutional feature maps for each image
which are extracted from **res4f_relu** layer of a ResNet-50 trained on ImageNet:
(The `fp16` suffix means that the `dtype` is `float16`.)

```
>> valfeats = numpy.load('flickr30k_ResNets50_blck4_val.fp16.npy')
>> valfeats.shape
(1014, 196, 1024)
# 1014: n_samples
#  196: flattened 14x14 into 196 for convenience
# 1024: n_feature_maps
```

For more information about the image features, please refer to:

```
@article{caglayan2016does,
  title={Does Multimodality Help Human and Machine for Translation and Image Captioning?},
  author={Caglayan, Ozan and Aransa, Walid and Wang, Yaxing and Masana, Marc and Garc{\'\i}a-Mart{\'\i}nez, Mercedes and Bougares, Fethi and Barrault, Lo{\"\i}c and van de Weijer, Joost},
  journal={arXiv preprint arXiv:1605.09186},
  year={2016}
}
```

## Factored Neural Machine Translation system

The Factored NMT models defined by basefnmt.py are based on the NMT architecture and extended to be able to generate several output symbols at the same time (Figure http://www-lium.univ-lemans.fr/~garcia/fnmt_archi.pdf).

Folder `ted-factors` contains examples of how to use this system.

### Citation:

If you use fnmt system in your work, please cite the following:

```
@inproceedings{garcia-martinez2016fnmt,
  title={Factored Neural Machine Translation Architectures},
  author={Garc{\'\i}a-Mart{\'\i}nez, Mercedes and Barrault, Lo{\"\i}c and Bougares, Fethi},
  booktitle={Proceedings of the International Workshop on Spoken Language Translation (IWSLT)},
  year={2016},
  url={'http://workshop2016.iwslt.org/downloads/IWSLT_2016_paper_2.pdf'}
}
```

Contact: Mercedes.Garcia_Martinez@univ-lemans.fr.
