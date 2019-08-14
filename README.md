# Neural Graph to Sequence Model

This repository contains the code for our paper [A Graph-to-Sequence Model for AMR-to-Text Generation](https://arxiv.org/abs/1805.02473) in ACL 2018

The code is developed under TensorFlow 1.4.1. 
We shared our pretrained model along this repository. 
Due to the compitibility reason of TensorFlow, it may not be loaded by some lower version (such as 1.0.0).

Please create issues if there are any questions! This can make things more tractable. 

## Update logs

### RESULTS ON WebNLG datasets (Aug. 4th, 2019)
Following the same setting (such as data preprocessing) of [Marcheggiani and Perez-Beltrachini (INLG 2019)](https://www.aclweb.org/anthology/W18-6501), our model achieves a BLEU score of *64.2*.

### Be careful about your tokenizer (Feb. 27th, 2019)
We use the [PTB_tokenizer](https://nlp.stanford.edu/software/tokenizer.shtml) from Stanford corenlp to preprocess our data. If you plan to use our pretrained model, please be careful on the tokenizer you use. 
Also, be careful to keep the words cased during preprocessing, as the PTB_tokenizer is sensitive to that.

### Release of 2M automatically parsed data (Dec. 4th, 2018)
We release our [2M sentences with their automatically parsed AMRs](https://www.cs.rochester.edu/~lsong10/downloads/2m.json.gz) to the public.

## About AMR

AMR is a graph-based semantic formalism, which can unified representations for several sentences of the same meaning.
Comparing with other structures, such as dependency and semantic roles, the AMR graphs have several key differences:
* AMRs only focus on concepts and their relations, so no function words are included. Actually the edge labels serve the role of function words.
* Inflections are dropped when converting a noun, a verb or named entity into a AMR concept. Sometimes a synonym is used instead of the original word. This makes more unified AMRs so that each AMR graph can represent more sentences.
* Relation tags (edge labels) are predefined and are not extracted from text (like the way OpenIE does).
More details are in the official AMR page [AMR website@ISI](https://amr.isi.edu/download.html), where
you can download the public-available AMR bank: [little prince](https://amr.isi.edu/download/amr-bank-struct-v1.6.txt).
Try it for fun!

## Data precrocessing
The [data loader](./src_g2s/G2S_data_stream.py) of our model requires simplified AMR graphs where variable tags, sense tags and quotes are removed. For example, the following AMR
```
(d / describe-01 
    :ARG0 (p / person 
        :name (n / name 
            :op1 "Ryan")) 
    :ARG1 p 
    :ARG2 genius)
```
need to be simplified as
```
describe :arg0 ( person :name ( name :op1 ryan )  )  :arg1 person :arg2 genius
```
before being consumed by our model.


We provide our scripts for AMR simplification.
First, you need to make each AMR into a single line, where our released [script](./AMR_multiline_to_singleline.py) may serve your goal (You may need to slightly modify it).
Second, to simplify the single-line AMRs, we release our tool that can be downloaded [here](https://www.cs.rochester.edu/~lsong10/downloads/amr_simplifier.tgz).
It is adapted from the [NeuralAMR](https://github.com/sinantie/NeuralAmr) system.
To run our simplifier on a file ```demo.amr```, simply execute
```
./anonDeAnon_java.sh anonymizeAmrFull true demo.amr
```
and it will output the simplified AMRs into ```demo.amr.anonymized```.
Please note that our simplifier *does not* do anonymization.
The resulting filename contains the 'anonymized' string because the original NeuralAMR creates the suffix.


Another alternative is to write your own data loading code according to the format of your own AMR data. 


### Input data format
After simplifying your AMRs, you can merge them with the corresponding sentences into a JSON file. 
The JSON file is the actual input to the system.
Its format is shown with the following sample:
```
[{"amr": "describe :arg0 ( person :name ( name :op1 ryan )  )  :arg1 person :arg2 genius",
"sent": "ryan 's description of himself : a genius .",
"id": "demo"}]
```
In general, the JSON file contains a list of instances, and each instance is a dictionary with fields of "amr", "sent" and "id"(optional).

### Vocabulary extraction
After having the JSON files, you can extract vocabularies with our released scripts in the [./data/](./data/) directory.
We also encourage you to write your own scripts.

## Training

First, modify the PYTHONPATH within [train_g2s.sh](./train_g2s.sh) (for our graph-to-string model) or [train_s2s.sh](./train_s2s.sh) (for baseline). <br>
Second, modify config_g2s.json or config_s2s.json. You should pay attention to the field "suffix", which is an identifier of the model being trained and saved. We usually use the experiment setting, such as "bch20_lr1e3_l21e3", as the identifier. <br>
Finally, execute the corresponding script file, such as "./train_g2s.sh".

### Using large-scale automatic AMRs

In this setting, we follow [Konstas et al., (2017)](https://arxiv.org/abs/1704.08381) to take the large-scale automatic data as the training set, taking the original gold data as a finetune set. 
To perform training in this way, you need to add a new field "finetune_path" in your config file and point it to the gold data. Besides the oringinal "train_path" should point to the automatic data. 

For training on the gold data only, we use an initial learning rate of 1e-3 and L2 normalization of 1e-3. We then lower the learning rate to be 8e-4, 5e-4 and 2e-4 after a number of epoches. 

For training on both gold and automatic data, the initial learning rate and L2 normalization are 5e-4 and 1e-8. We also lower the learning rate during training. 

The idea of lowering learning rate was first introduced by Konstas et al., (2017).


## Decoding with a pretained model

Simply execute the corresponding decoding script with one argument being the identifier of the model you want to use.
For instance, you can execute "./decode_g2s.sh bch20_lr1e3_l21e3".
Please make sure you use the associated word vectors not others, because the pretrained model are *optimized* given the word vectors.

### Pretrained model

We release a pretrained model (and word vectors) using gold plus 2M automatically-parsed AMRs [here](https://www.cs.rochester.edu/~lsong10/downloads/model_silver_2m.tgz). With this model, we observed a BLEU of *33.6*, which is higher than our paper-reported number of 33.0. The pretrained model with only gold data is [here](https://www.cs.rochester.edu/~lsong10/downloads/model_gold.tgz). It reports a test BLEU score of 23.3.

## Cite
If you like our paper, please cite
```
@inproceedings{song2018graph,
  title={A Graph-to-Sequence Model for AMR-to-Text Generation},
  author={Song, Linfeng and Zhang, Yue and Wang, Zhiguo and Gildea, Daniel},
  booktitle={Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={1616--1626},
  year={2018}
}
```
