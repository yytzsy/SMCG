import re
import json
import argparse
import numpy as np
import nltk
import cPickle as pkl
import random

path = '../activitynet_data/'
anet_caption = json.load(open(path+'anet_caption.json'))
coco_caption = json.load(open(path+'coco_caption_only.json'))

all_template_num = 20
all_coco_caption_num = len(coco_caption)
for video_id in anet_caption:
    anet_caption[video_id]['template_captions'] = []
    anet_caption[video_id]['template_final_captions'] = []
    anet_caption[video_id]['template_pos'] = []
    anet_caption[video_id]['template_parse'] = []
    anet_caption[video_id]['template_raw_parse'] = []
    for cc in range(all_template_num):
        choose_id = str(random.randint(0,all_coco_caption_num-1))
        coco_content = coco_caption[choose_id]

        template_captions = coco_content['captions'][0]
        template_final_captions = coco_content['final_captions'][0]
        template_pos = coco_content['final_pos'][0]
        template_parse = coco_content['final_parse'][0]
        template_raw_parse = coco_content['raw_parse'][0]

        anet_caption[video_id]['template_captions'].append(template_captions)
        anet_caption[video_id]['template_final_captions'].append(template_final_captions)
        anet_caption[video_id]['template_pos'].append(template_pos)
        anet_caption[video_id]['template_parse'].append(template_parse)
        anet_caption[video_id]['template_raw_parse'].append(template_raw_parse)


json.dump(anet_caption, open(path+'anet_caption_with_template.json', 'w'))