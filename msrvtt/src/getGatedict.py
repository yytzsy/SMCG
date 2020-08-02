import json
import os
import numpy as np


samples = np.load('/DATA-NFS/yuanyitian/sentence_autoencoder/sentence_autoencoder/results/S2VTAttModel.npz')
samples = samples['predictions']
samples = samples.tolist()
gate_dict = {}
for k,sentence_id in enumerate(samples):
    content = samples[sentence_id][0]
    groundtruth_sentence = content['groundtruth_caption']
    predict_sentence = content['caption']
    gate = content['gate']
    gate_dict[groundtruth_sentence] = [gate,predict_sentence]
    print k
    
np.save(open('gate_dict.npy','w'),gate_dict)

