# SMCG
Code for the paper "Controllable Video Captioning with an Exemplar Sentence"


## Introduction
 
We investigate a novel and challenging task, namely controllable video captioning with an exemplar sentence. Formally, given a video and a syntactically valid exemplar sentence, the task aims to generate one caption which not only describes the semantic contents of the video, but also follows the syntactic form of the given exemplar sentence. In order to tackle such an exemplar-based video captioning task, we propose a novel Syntax Modulated Caption Generator (SMCG) incorporated in an encoder-decoder-reconstructor architecture.
![](https://github.com/yytzsy/SCDM/blob/master/task.PNG)

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
```
```
python fill_template.py
```
* Go to the '**./activitynet/process_activitynet_data/**' folder, and run:
```
python prepro_anetcoco_vocab_pos_parse.py
```
```
python fill_template.py
```

## Model Training and Testing
![](https://github.com/yytzsy/SCDM/blob/master/model.PNG)

* For the Charades-STA dataset, the proposed model and all its variant models are provided. For example, the proposed SCDM model implementation is in the '**./grounding/Charades-STA/src_SCDM**' folder, run:
```
python run_charades_scdm.py --task train
```
for model training, and run:
```
python run_charades_scdm.py --task test
```
for model testing. Other variant models are similar to train and test.

* For the TACoS and ActivityNet Captions dataset, we only provide the proposed SCDM model implementation in the '**./grounding/xxx/src_SCDM**' folder. The training and testing process are similar to the Charades-STA dataset.
* Please train our provided models from scratch, and you can reproduce the results in the paper (not exactly the same, but almost).

## Citation
```
@inproceedings{yuan2019semantic,
  title={Semantic Conditioned Dynamic Modulation for Temporal Sentence Grounding in Videos},
  author={Yuan, Yitian and Ma, Lin and Wang, Jingwen and Liu, Wei and Zhu, Wenwu},
  booktitle={Advances in Neural Information Processing Systems},
  pages={534--544},
  year={2019}
}
```
