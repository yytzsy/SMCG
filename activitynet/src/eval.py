import json
import os
import argparse
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from models import EncoderRNN, DecoderRNN, S2VTAttModel, S2VTModel, DecoderSyntaxRNN, DecoderVideoRNN
from dataloader import VideoDataset
import misc.utils as utils
from misc.cocoeval import suppress_stdout_stderr
from pandas.io.json import json_normalize
import numpy as np
from parse_tree import *
import logging
import time
import sys
reload(sys)
import cPickle as pkl
sys.setdefaultencoding('utf-8')
from evaluate_util import *
from misc.cocoeval import suppress_stdout_stderr, COCOScorer





def test(model, crit, dataset, vocab, opt, epoch):

    scorer = COCOScorer()

    info = json.load(open(opt["info_json"]))
    ix_to_word = info['ix_to_word']
    video_caption = json.load(open(opt['caption_json']))
    gts = pkl.load(open(opt['all_cap_info']))

    model.eval()
    loader = DataLoader(dataset, batch_size=1, shuffle=False)


    results = []
    output_dict = {}
    samples = {}
    extend_gts = {}


    sample_num = 0
    for data in loader:

        print os.path.abspath('./').split('/')[-1]
        print sample_num
        sample_num +=1

        # forward the model to get loss
        video_ids = data['video_ids']
        fc_feats = data['fc_feats'].cuda() #(b,40,dim) b=1
        labels = data['labels'].cuda() #(b,28) b=1
        template_labels = data['template_labels'].cuda() #(b,20,28)
        template_parse_labels = data['template_parse_labels'].cuda() #(b,20,120) b=1


        groundtruth_sentence_list = video_caption[video_ids[0]]['captions']
        caption_gts_num = len(groundtruth_sentence_list)


        template_sentence_list = []
        template_labels_data = template_labels.cpu().numpy()
        template_gts_num = np.shape(template_labels_data)[1]
        jj = 0
        while jj < template_gts_num:
            sentence_label = template_labels_data[0,jj,:]
            content = ''
            for iid in sentence_label:
                content = content + ' ' + ix_to_word[str(iid)]
                if ix_to_word[str(iid)] == '<eos>':
                    break
            content = content.strip()
            template_sentence_list.append(content)
            jj+=1


        fc_feats = np.tile(fc_feats.cpu().numpy(),[template_gts_num,1,1])
        labels = np.tile(labels.cpu().numpy(),[template_gts_num,1])
        template_parse_labels = np.squeeze(template_parse_labels.cpu().numpy())

        fc_feats = torch.from_numpy(fc_feats).cuda()
        labels = torch.from_numpy(labels).cuda()
        template_parse_labels = torch.from_numpy(template_parse_labels).cuda()


        new_video_ids = []
        if len(video_ids) == 1:
            for kk in range(template_gts_num):
                new_video_ids.append(video_ids[0])
        else:
            new_video_ids = video_ids
        
        with torch.no_grad():
            seq_probs, seq_preds, _, _, _ = model(fc_feats, labels, template_parse_labels, mode='inference', opt=opt)
        sents = utils.decode_sequence(vocab, seq_preds)

        for k, sent in enumerate(sents):
            video_id = new_video_ids[k]
            extend_video_id = video_id+'_'+str(k)
            samples[extend_video_id] = [{'image_id': video_id, 'caption': sent}]
            extend_gts[extend_video_id] = gts[video_id]

    print len(extend_gts)
    print len(samples)
    with suppress_stdout_stderr():
        valid_score = scorer.score(extend_gts, samples, samples.keys())
    results.append(valid_score)
    print(valid_score)


    if not os.path.exists(opt["results_path"]):
        os.makedirs(opt["results_path"])
    with open(os.path.join(opt["results_path"], "caption_scores.txt"), 'a') as scores_table:
        scores_table.write("Epoch "+str(epoch)+": ")
        scores_table.write(json.dumps(results[0])+"\n")





def main(opt):


    dataset = VideoDataset(opt, "test")
    opt["vocab_size"] = dataset.get_vocab_size()
    opt["parse_size"] = dataset.get_parse_vocab_size()


    opt["seq_length"] = dataset.max_len
    if opt["model"] == 'S2VTModel':
        model = S2VTModel(opt["vocab_size"], opt["max_len"], opt["dim_hidden"], opt["dim_word"],
                          rnn_dropout_p=opt["rnn_dropout_p"]).cuda()
    elif opt["model"] == "S2VTAttModel":
        encoder = EncoderRNN(
            opt["dim_parse"],
            opt["parse_size"],
            opt["dim_vid"],
            opt["dim_hidden"],
            n_layers=opt["num_layers"],
            bidirectional=opt["bidirectional"],
            input_dropout_p=opt["input_dropout_p"],
            rnn_cell=opt['rnn_type'],
            rnn_dropout_p=opt["rnn_dropout_p"],
            embedding_pretrained_weights = None
            )
        decoder = DecoderRNN(
            opt["vocab_size"],
            opt["max_len"],
            opt["dim_hidden"],
            opt["dim_word"],
            opt["dim_parse"],
            n_layers=opt["num_layers"],
            input_dropout_p=opt["input_dropout_p"],
            rnn_cell=opt['decode_rnn_type'],
            rnn_dropout_p=opt["rnn_dropout_p"],
            bidirectional=opt["bidirectional"],
            w_dropout_p=opt["w_dropout_p"],
            embedding_pretrained_weights = None
            )
        decoder_syntax = DecoderSyntaxRNN(
            opt['parse_size'],
            opt['parse_max_len'],
            opt['dim_hidden'],
            opt['dim_parse'],
            n_layers=opt["num_layers"],
            rnn_cell=opt['decode_rnn_type'],
            bidirectional=opt["bidirectional"],
            input_dropout_p=opt["input_dropout_p"],
            rnn_dropout_p=opt["rnn_dropout_p"],
            )
        decoder_video = DecoderVideoRNN(
            opt['video_max_len'],
            opt['dim_hidden'],
            opt['dim_vid'],
            n_layers=opt["num_layers"],
            bidirectional=opt["bidirectional"],
            input_dropout_p=opt["input_dropout_p"],
            rnn_dropout_p=opt["rnn_dropout_p"],
            )
        model = S2VTAttModel(encoder, decoder, decoder_syntax, decoder_video, opt['dim_word'], opt['dim_hidden']).cuda()
    #model = nn.DataParallel(model)
    # Setup the model
    model.load_state_dict(torch.load(opt["saved_model"]))
    epoch_num = int(opt["saved_model"].split('_')[-1][:-4])

    crit = utils.LanguageModelCriterion()

    test(model, crit, dataset, dataset.get_vocab(), opt, epoch_num)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--recover_opt', type=str, required=True,
                        help='recover train opts from saved opt_json')
    parser.add_argument('--saved_model', type=str, default='./save/',
                        help='path to saved model to evaluate')

    parser.add_argument('--dump_json', type=int, default=1,
                        help='Dump json with predictions into vis folder? (1=yes,0=no)')
    parser.add_argument('--results_path', type=str, default='./results/')
    parser.add_argument('--dump_path', type=int, default=0,
                        help='Write image paths along with predictions into vis json? (1=yes,0=no)')
    parser.add_argument('--gpu', type=str, default='0',
                        help='gpu device number')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='minibatch size')
    parser.add_argument('--sample_max', type=int, default=1,
                        help='0/1. whether sample max probs  to get next word in inference stage')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--beam_size', type=int, default=1,
                        help='used when sample_max = 1. Usually 2 or 3 works well.')

    args = parser.parse_args()
    args = vars((args))
    opt = json.load(open(args["recover_opt"]))
    for k, v in args.items():
        opt[k] = v
    os.environ['CUDA_VISIBLE_DEVICES'] = opt["gpu"]
    main(opt)
