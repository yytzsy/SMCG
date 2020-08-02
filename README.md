# SMCG
Code for the paper "Controllable Video Captioning with an Exemplar Sentence"


## Introduction
 
We investigate a novel and challenging task, namely controllable video captioning with an exemplar sentence. Formally, given a video and a syntactically valid exemplar sentence, the task aims to generate one caption which not only describes the semantic contents of the video, but also follows the syntactic form of the given exemplar sentence. In order to tackle such an exemplar-based video captioning task, we propose a novel Syntax Modulated Caption Generator (SMCG) incorporated in an encoder-decoder-reconstructor architecture.
![](https://github.com/yytzsy/SMCG/blob/master/task.jpg)

## Dependency

* python 2.7.2
* torch 1.1.0
* java openjdk version "10.0.2" 2018-07-17

## Download Features and Preprocess Data

For the MSRVTT dataset, please download the following files into the '**./msrvtt/msrvtt_data**' folder:
* MSRVTT caption info: [videodatainfo_2016.json](https://drive.google.com/file/d/1P-kfWOQoHzSxd8vNpogNGyx8Jc4TKj4E/view?usp=sharing),
* MSRVTT captions and their sentence parse trees: [msrvtt_all_sentence_parse_dict.pkl](https://drive.google.com/file/d/1P-kfWOQoHzSxd8vNpogNGyx8Jc4TKj4E/view?usp=sharing),
* Collected exemplar sentences and their parse trees: [coco_filter_parse_dict.pkl](https://drive.google.com/file/d/1P-kfWOQoHzSxd8vNpogNGyx8Jc4TKj4E/view?usp=sharing),
* Video features: [msrvtt_incepRes_rgb_feats.hdf5](https://drive.google.com/file/d/1P-kfWOQoHzSxd8vNpogNGyx8Jc4TKj4E/view?usp=sharing),
* Glove word embeddings: [glove.840B.300d.zip](http://nlp.stanford.edu/data/glove.840B.300d.zip).


For the ActivityNet Captionsd dataset, please download the following files into the '**./activitynet/activitynet_data**' folder:
* ActivityNet caption info: [CAP.pkl](https://drive.google.com/file/d/1P-kfWOQoHzSxd8vNpogNGyx8Jc4TKj4E/view?usp=sharing),
* ActivityNet captions and their sentence parse trees: [anet_parse_dict.pkl](https://drive.google.com/file/d/1P-kfWOQoHzSxd8vNpogNGyx8Jc4TKj4E/view?usp=sharing),
* Collected exemplar sentences and their parse trees: [coco_filter_parse_dict.pkl](https://drive.google.com/file/d/1P-kfWOQoHzSxd8vNpogNGyx8Jc4TKj4E/view?usp=sharing),
* Video features: [anet_new_inception_resnet_feats.hdf5](https://drive.google.com/file/d/1P-kfWOQoHzSxd8vNpogNGyx8Jc4TKj4E/view?usp=sharing),
* Glove word embeddings: [glove.840B.300d.zip](http://nlp.stanford.edu/data/glove.840B.300d.zip).


## Data Preprocessing


* Go to the '**./msrvtt/process_msrvtt_data/**' folder, and run:
```
python prepro_vocab_parse_pos.py
python fill_template.py
```
* Go to the '**./activitynet/process_activitynet_data/**' folder, and run:
```
python prepro_anetcoco_vocab_pos_parse.py
python fill_template.py
```

## Model Training and Testing
![](https://github.com/yytzsy/SMCG/blob/master/frame.jpg)

* For the MSRVTT dataset, please go to the '**./msrvtt/src/**' folder, and train the model by:
```
python train.py --gpu xx
```
* For model inference and evaluation, run:
```
bash eval.sh 
bash control.sh 
```
* Note: 'eval.sh' is used to evaluate the generated exemplar-based captions with conventional captioning metrics. 'control.sh' is used to compare the generated exemplar-based captions with the provided exemplar captions from the syntactic aspect, i.e., compute the edit distance between their parse trees.


* For the ActivityNet Captions dataset, please go to the '**./activitynet/src/**' folder, and train/test the model as on the MSRVTT dataset.

## Citation
```
@inproceedings{yuan2020Control,
  title={Controllable Video Captioning with an Exemplar Sentence},
  author={Yuan, Yitian and Ma, Lin and Wang, Jingwen and Zhu, Wenwu},
  booktitle={the 28th ACM International Conference on Multimedia (MM ’20)},
  year={2020}
}
```
