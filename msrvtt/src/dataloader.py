import json

import random
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import h5py

class VideoDataset(Dataset):

    def get_vocab_size(self):
        return len(self.get_vocab())

    def get_vocab(self):
        return self.ix_to_word

    def get_seq_length(self):
        return self.seq_length

    def get_parse_vocab_size(self):
        return len(self.get_parse_vocab())

    def get_parse_vocab(self):
        return self.parse_ix_to_word

    def __init__(self, opt, mode):
        super(VideoDataset, self).__init__()
        self.mode = mode  # to load train/val/test data

        info_extend = json.load(open(opt["info_json_extend"]))
        self.ix_to_word = info_extend['ix_to_word']
        self.word_to_ix = info_extend['word_to_ix']
        print('vocab size is ', len(self.ix_to_word))

        self.parse_ix_to_word = info_extend['parse_ix_to_word']
        self.parse_word_to_ix = info_extend['parse_word_to_ix']
        print('parse vocab size is ', len(self.parse_ix_to_word))

        # load the json file which contains information about the dataset
        info = json.load(open(opt["info_json"]))
        self.splits = info['videos']
        print('number of train videos: ', len(self.splits['train']))
        print('number of val videos: ', len(self.splits['val']))
        print('number of test videos: ', len(self.splits['test']))

        self.captions = json.load(open(opt["caption_json"]))
        
        self.all_features = h5py.File(opt["features_inception_resnet_path"],'r')
        # load in the sequence data
        self.max_len = opt["max_len"]
        self.parse_max_len = opt["parse_max_len"]
        print('max sequence length in data is', self.max_len)
        print('max sequence length in parse is', self.parse_max_len)




    def __getitem__(self, ix):
        """This function returns a tuple that is further passed to collate_fn
        """
        # which part of data to load
        if self.mode == 'val':
            ix += len(self.splits['train'])
        elif self.mode == 'test':
            ix = ix + len(self.splits['train']) + len(self.splits['val'])
        
        fc_feat = np.array(self.all_features['vid%i'% (ix+1)])

        ############################################ get random video ##########################################

        random_ix = random.randint(0,len(self.splits[self.mode])-1)
        if self.mode == 'val':
            random_ix += len(self.splits['train'])
        elif self.mode == 'test':
            random_ix = random_ix + len(self.splits['train']) + len(self.splits['val'])
        random_fc_feat = np.array(self.all_features['vid%i'% (random_ix+1)])
        
        #################################################################################################################



        captions = self.captions['video%i'%(ix)]['final_captions']
        gts = np.zeros((len(captions), self.max_len))
        for i, cap in enumerate(captions):
            if len(cap) > self.max_len:
                cap = cap[:self.max_len]
                cap[-1] = '<eos>'
            for j, w in enumerate(cap):
                gts[i, j] = self.word_to_ix[w]


        captions_parse = self.captions['video%i'%(ix)]['final_parse']
        gts_parse = np.zeros((len(captions_parse), self.parse_max_len))
        for i, parse in enumerate(captions_parse):
            if len(parse) > self.parse_max_len:
                parse = parse[:self.parse_max_len]
                parse[-1] = '<eos>'
            for j, w in enumerate(parse):
                gts_parse[i, j] = self.parse_word_to_ix[w]


        # random select a caption for this video
        label = np.zeros(self.max_len)
        label_parse = np.zeros(self.parse_max_len,)
        mask = np.zeros(self.max_len)
        mask_parse = np.zeros(self.parse_max_len,)
        count = 0
        while np.sum(label_parse) == 0 and count < 20:
            cap_ix = random.randint(0, len(captions) - 1)
            label = gts[cap_ix]
            label_parse = gts_parse[cap_ix]
            count += 1
        non_zero = (label == 0).nonzero()
        mask[:int(non_zero[0][0]) + 1] = 1
        parse_non_zero = (label_parse == 0).nonzero()
        mask_parse[:int(parse_non_zero[0][0]) + 1] = 1

        groundtruth_sentence_str = ''
        for item in captions[cap_ix]:
            groundtruth_sentence_str = groundtruth_sentence_str + ' ' + item
        groundtruth_sentence_str = groundtruth_sentence_str.strip()
        #################################### get template sentence and parse #############################
 
        template_final_captions = self.captions['video%i'%(ix)]['template_final_captions']
        template_labels = np.zeros((len(template_final_captions),self.max_len))
        for i, cap in enumerate(template_final_captions):
            if len(cap) > self.max_len:
                cap = cap[:self.max_len]
                cap[-1] = '<eos>'
            for j, w in enumerate(cap):
                template_labels[i, j] = self.word_to_ix[w]

        template_parse = self.captions['video%i'%(ix)]['template_parse']
        template_parse_labels = np.zeros((len(template_parse),self.parse_max_len))
        for i, parse in enumerate(template_parse):
            if len(parse) > self.parse_max_len:
                parse = parse[:self.parse_max_len]
                parse[-1] = '<eos>'
            for j, w in enumerate(parse):
                template_parse_labels[i, j] = self.parse_word_to_ix[w]


        data = {}
        data['fc_feats'] = torch.from_numpy(fc_feat).type(torch.FloatTensor)
        data['random_fc_feat'] = torch.from_numpy(random_fc_feat).type(torch.FloatTensor)
        data['labels'] = torch.from_numpy(label).type(torch.LongTensor)
        data['label_parse'] = torch.from_numpy(label_parse).long()
        data['masks'] = torch.from_numpy(mask).type(torch.FloatTensor)
        data['mask_parse'] = torch.from_numpy(mask_parse).type(torch.FloatTensor)
        data['gts'] = torch.from_numpy(gts).long()
        data['gts_parse'] = torch.from_numpy(gts_parse).long()
        data['video_ids'] = 'video%i'%(ix)
        data['groundtruth_sentence_str'] = groundtruth_sentence_str
        
        data['template_labels'] = torch.from_numpy(template_labels).type(torch.LongTensor)
        data['template_parse_labels'] = torch.from_numpy(template_parse_labels).type(torch.LongTensor)

        return data

    def __len__(self):
        return len(self.splits[self.mode])
