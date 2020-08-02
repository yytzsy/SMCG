import re
import json
import argparse
import numpy as np
import nltk
import cPickle as pkl

def build_vocab(vids, coco_extend_sentence, params):
    print 'start build vocabulary'
    count_thr = params['word_count_threshold']

    counts = {}
    for vid, caps in vids.items():
        for cap in caps['captions']:
            cap = cap.lower()
            ws = re.sub(r'[.!,;?]', ' ', cap).split()
            for w in ws:
                counts[w] = counts.get(w, 0) + 1

    extra_counts = {}
    for cap in coco_extend_sentence:
        cap = cap.lower()
        ws = re.sub(r'[.!,;?]', ' ', cap).split()
        for w in ws:
            extra_counts[w] = extra_counts.get(w,0) + 1

    total_words = sum(counts.values())
    bad_words = [w for w, n in counts.items() if n <= count_thr]
    vocab = [w for w, n in counts.items() if n > count_thr]
    bad_count = sum(counts[w] for w in bad_words)
    print('number of bad words: %d/%d = %.2f%%' %
          (len(bad_words), len(counts), len(bad_words) * 100.0 / len(counts)))
    print('number of words in vocab would be %d' % (len(vocab), ))
    print('number of UNKs: %d/%d = %.2f%%' %
          (bad_count, total_words, bad_count * 100.0 / total_words))
    # lets now produce the final annotations
    if bad_count > 0:
        # additional special UNK token we will use below to map infrequent words to
        print('inserting the special UNK token')
        vocab.append('<UNK>')

    extra_vocab = [w for w, n in extra_counts.items() if n > count_thr]
    vocab = vocab + extra_vocab
    print len(vocab)
    vocab = list(set(vocab))
    print len(vocab)

    return vocab,counts,extra_counts


def get_pos_vocab(video_caption_msrvtt, video_caption_coco):
    counts = {}

    for id_num in video_caption_msrvtt:
        pos_lists = video_caption_msrvtt[id_num]['final_pos']
        for pos_list in pos_lists:
            for pos in pos_list:
                if pos != '<eos>' and pos != '<sos>':
                    counts[pos] = counts.get(pos, 0) + 1

    for id_num in video_caption_coco:
        pos_list = video_caption_coco[id_num]['final_pos']
        pos_list = pos_list[0]
        for pos in pos_list:
            if pos != '<eos>' and pos != '<sos>':
                counts[pos] = counts.get(pos, 0) + 1

    pos_vocab = [w for w, n in counts.items() if n > 1]
    pos_itow = {i + 2: w for i, w in enumerate(pos_vocab)}
    pos_wtoi = {w: i + 2 for i, w in enumerate(pos_vocab)}  # inverse table
    pos_wtoi['<eos>'] = 0
    pos_itow[0] = '<eos>'
    pos_wtoi['<sos>'] = 1
    pos_itow[1] = '<sos>'
    print len(pos_vocab)
    print len(pos_itow)
    print len(pos_wtoi)
    return pos_vocab, pos_itow, pos_wtoi



def get_parse_vocab(video_caption_msrvtt, video_caption_coco):
    counts = {}

    for id_num in video_caption_msrvtt:
        parse_lists = video_caption_msrvtt[id_num]['final_parse']
        for parse_list in parse_lists:
            for parse in parse_list:
                if parse != '<sos>' and parse != '<eos>':
                    counts[parse] = counts.get(parse, 0) + 1

    for id_num in video_caption_coco:
        parse_list = video_caption_coco[id_num]['final_parse']
        parse_list = parse_list[0]
        for parse in parse_list:
            if parse != '<sos>' and parse != '<eos>':
                counts[parse] = counts.get(parse, 0) + 1


    parse_vocab = [w for w, n in counts.items() if n > 1]
    parse_itow = {i + 2: w for i, w in enumerate(parse_vocab)}
    parse_wtoi = {w: i + 2 for i, w in enumerate(parse_vocab)}  # inverse table
    parse_wtoi['<eos>'] = 0
    parse_itow[0] = '<eos>'
    parse_wtoi['<sos>'] = 1
    parse_itow[1] = '<sos>'
    print len(parse_vocab)
    print len(parse_itow)
    print len(parse_wtoi)
    return parse_vocab, parse_itow, parse_wtoi



def main(params):

    count_thr = params['word_count_threshold']
    ##############################  for create vocabulary coco & msrvtt ##################################
    # origin_msrvtt_video_caption_data
    captions = json.load(open(params['input_json'], 'r'))['sentences']
    video_caption = {}
    sentence_id = 0
    for i in captions:
        video_caption[sentence_id] = {'captions': []}
        tmp = i['caption']
        tmp = tmp.lower()
        video_caption[sentence_id]['captions'].append(tmp)
        sentence_id +=1
    
    # create the vocab
    coco_parse_dict = pkl.load(open(params['extend_caption_dict']))
    coco_extend_sentence = coco_parse_dict.keys()
    vocab,counts,extra_counts = build_vocab(video_caption, coco_extend_sentence, params)
    itow = {i + 2: w for i, w in enumerate(vocab)}
    wtoi = {w: i + 2 for i, w in enumerate(vocab)}  # inverse table
    wtoi['<eos>'] = 0
    itow[0] = '<eos>'
    wtoi['<sos>'] = 1
    itow[1] = '<sos>'

    print 'word dict build'
    print len(wtoi)
    print len(itow)


    ######################## for create msrvtt caption json ################################
    msrvtt_parse_dict = pkl.load(open(params['msrvtt_parse_dict']))
    videos = json.load(open(params['input_json'], 'r'))['sentences']
    msrvtt_video_caption = {}
    for i in videos:
        if i['video_id'] not in msrvtt_video_caption.keys():
            msrvtt_video_caption[i['video_id']] = {'captions': []}
        tmp = i['caption']
        tmp = tmp.lower()
        msrvtt_video_caption[i['video_id']]['captions'].append(tmp)

    parse_bad_count = 0
    all_count = 0
    # all_msrvtt_sentence_file = open('all_msrvtt_sentence.txt','w')
    for vid, caps in msrvtt_video_caption.items():
        caps = caps['captions']
        msrvtt_video_caption[vid]['final_captions'] = []
        msrvtt_video_caption[vid]['final_pos'] = []
        msrvtt_video_caption[vid]['final_parse'] = []
        msrvtt_video_caption[vid]['raw_parse'] = []
        for cap in caps:
            cap = cap.lower()
            ws = re.sub(r'[.!,;?]', ' ', cap).split()
            caption = ['<sos>'] + [w if counts.get(w, 0) > count_thr else '<UNK>' for w in ws] + ['<eos>']
            msrvtt_video_caption[vid]['final_captions'].append(caption)

            tmp_str_list = [w if counts.get(w, 0) > count_thr else '<UNK>' for w in ws]
            pos_list = nltk.pos_tag(tmp_str_list)
            tmp = ['<sos>'] + [pos[1] for pos in pos_list] + ['<eos>']
            msrvtt_video_caption[vid]['final_pos'].append(tmp)


            new_cap = cap.strip()
            if new_cap in msrvtt_parse_dict:
                msrvtt_video_caption[vid]['final_parse'].append(['<sos>']+msrvtt_parse_dict[new_cap]['process_parse_list']+['<eos>'])
                msrvtt_video_caption[vid]['raw_parse'].append(msrvtt_parse_dict[new_cap]['raw_parse'])
            else:
                msrvtt_video_caption[vid]['final_parse'].append([])
                msrvtt_video_caption[vid]['raw_parse'].append([])
                parse_bad_count +=1

            # all_msrvtt_sentence_file.write(new_cap+'\n')
            all_count +=1

    print('msrvtt number of parse_bad_count: %d' % (parse_bad_count, )) 
    print('msrvtt number of all_count: %d' % (all_count, )) 
    json.dump(msrvtt_video_caption, open(params['msrvtt_caption_json'], 'w'))


    ######################## for create coco caption json ################################
    sentence_id = 0
    coco_video_caption = {}
    for sentence in coco_extend_sentence:
        sentence = sentence.lower()
        ws = re.sub(r'[.!,;?]', ' ', sentence).split()
        tmp_str_list = [w if extra_counts.get(w, 0) > count_thr else '<UNK>' for w in ws]
        if tmp_str_list != ['<UNK>']:
            coco_video_caption[sentence_id] = {'captions': [], 'final_captions': [], 'final_pos': [], 'final_parse': [], 'raw_parse': []}
            coco_video_caption[sentence_id]['captions'].append(sentence)
            coco_video_caption[sentence_id]['final_captions'].append(['<sos>'] + tmp_str_list + ['<eos>'])

            parse_content = coco_parse_dict[sentence]
            coco_video_caption[sentence_id]['final_parse'].append(['<sos>']+parse_content['process_parse_list']+['<eos>'])
            coco_video_caption[sentence_id]['raw_parse'].append(parse_content['raw_parse'])

            pos_list = nltk.pos_tag(tmp_str_list)
            tmp = ['<sos>'] + [pos[1] for pos in pos_list] + ['<eos>']
            coco_video_caption[sentence_id]['final_pos'].append(tmp)

            sentence_id +=1
    coco_all_sentence_num = sentence_id
    print 'coco_all_sentences_num = '+str(coco_all_sentence_num)

    json.dump(coco_video_caption, open(params['coco_caption_json'], 'w'))


    ###################### output info ############################
    pos_vocab, pos_itow, pos_wtoi = get_pos_vocab(msrvtt_video_caption,coco_video_caption)
    parse_vocab, parse_itow, parse_wtoi = get_parse_vocab(msrvtt_video_caption,coco_video_caption)

    out = {}
    out['ix_to_word'] = itow
    out['word_to_ix'] = wtoi
    out['pos_ix_to_word'] = pos_itow
    out['pos_word_to_ix'] = pos_wtoi
    out['parse_ix_to_word'] = parse_itow
    out['parse_word_to_ix'] = parse_wtoi
    
    json.dump(out, open(params['info_json'], 'w'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # output
    parser.add_argument('--info_json', default='../msrvtt_data/msrvtt_coco_info_extend.json', help='info about iw2word and word2ix')
    parser.add_argument('--msrvtt_caption_json', default='../msrvtt_data/msrvtt_caption.json', help='caption json file')
    parser.add_argument('--coco_caption_json', default='../msrvtt_data/coco_caption.json', help='caption json file')
    parser.add_argument('--word_count_threshold', default=1, type=int,help='only words that occur more than this number of times will be put in vocab')
    
    # input
    parser.add_argument('--input_json', type=str, default='../msrvtt_data/videodatainfo_2016.json',help='msr_vtt videoinfo json')
    parser.add_argument('--msrvtt_parse_dict', default='../msrvtt_data/msrvtt_all_sentence_parse_dict.pkl', help='pkl file')
    parser.add_argument('--extend_caption_dict', default='../msrvtt_data/coco_filter_parse_dict.pkl', help='pkl file')

    args = parser.parse_args()
    params = vars(args)  # convert to ordinary dict
    main(params)
