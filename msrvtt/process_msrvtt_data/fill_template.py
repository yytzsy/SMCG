import re
import json
import argparse
import numpy as np
import nltk
import cPickle as pkl
import random

path = '../msrvtt_data/'
msrvtt_caption = json.load(open(path+'msrvtt_caption.json'))
coco_caption = json.load(open(path+'coco_caption.json'))

all_template_num = 20
all_coco_caption_num = len(coco_caption)
for video_id in msrvtt_caption:
    msrvtt_caption[video_id]['template_captions'] = []
    msrvtt_caption[video_id]['template_final_captions'] = []
    msrvtt_caption[video_id]['template_pos'] = []
    msrvtt_caption[video_id]['template_parse'] = []
    msrvtt_caption[video_id]['template_raw_parse'] = []
    for cc in range(all_template_num):
        choose_id = str(random.randint(0,all_coco_caption_num-1))
        coco_content = coco_caption[choose_id]

        template_captions = coco_content['captions'][0]
        template_final_captions = coco_content['final_captions'][0]
        template_pos = coco_content['final_pos'][0]
        template_parse = coco_content['final_parse'][0]
        template_raw_parse = coco_content['raw_parse'][0]

        msrvtt_caption[video_id]['template_captions'].append(template_captions)
        msrvtt_caption[video_id]['template_final_captions'].append(template_final_captions)
        msrvtt_caption[video_id]['template_pos'].append(template_pos)
        msrvtt_caption[video_id]['template_parse'].append(template_parse)
        msrvtt_caption[video_id]['template_raw_parse'].append(template_raw_parse)


json.dump(msrvtt_caption, open(path+'msrvtt_caption_with_template.json', 'w'))