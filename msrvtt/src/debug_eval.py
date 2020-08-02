import json
import os
import argparse
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from models import EncoderRNN, DecoderRNN, S2VTAttModel, S2VTModel
from dataloader import SentenceDataset
import misc.utils as utils
from misc.cocoeval import suppress_stdout_stderr, COCOScorer

from pandas.io.json import json_normalize

import numpy as np
from parse_tree import *


import sys
reload(sys)
sys.setdefaultencoding('utf-8')


def convert_data_to_coco_scorer_format(all_captions):
    gts = {}
    all_caption_num = len(all_captions)
    for ix in range(all_caption_num):
        gts['sentence%i'%(ix)] = []
        for idx,cap in enumerate(all_captions['%i'%(ix)]['captions']):
            gts['sentence%i'%(ix)].append({'image_id':'sentence%i'%(ix), 'cap_id':idx, 'caption':cap})
    return gts

def test(model, crit, dataset, vocab, opt):

    model.eval()
    loader = DataLoader(dataset, batch_size=opt["batch_size"], shuffle=True)
    scorer = COCOScorer()

    all_captions = json.load(open(opt["caption_json"]))
    gts = convert_data_to_coco_scorer_format(all_captions)

    samples = np.load('/DATA-NFS/yuanyitian/sentence_autoencoder/sentence_autoencoder/results/S2VTAttModel.npz')
    samples = samples['predictions'].tolist()


    valid_score = scorer.score(gts, samples, samples.keys())
    print(valid_score)


def main(opt):
    dataset = SentenceDataset(opt, "test")
    opt["vocab_size"] = dataset.get_vocab_size()

    opt["seq_length"] = dataset.max_len
    if opt["model"] == 'S2VTModel':
        model = S2VTModel(opt["vocab_size"], opt["max_len"], opt["dim_hidden"], opt["dim_word"],
                          rnn_dropout_p=opt["rnn_dropout_p"]).cuda()
    elif opt["model"] == "S2VTAttModel":
        encoder = EncoderRNN(
            opt["vocab_size"],
            opt["dim_word"],
            opt["dim_hidden"],
            n_layers=opt["num_layers"],
            bidirectional=opt["bidirectional"],
            input_dropout_p=opt["input_dropout_p"],
            rnn_cell=opt['rnn_type'],
            rnn_dropout_p=opt["rnn_dropout_p"])
        decoder = DecoderRNN(
            opt["vocab_size"],
            opt["max_len"],
            opt["dim_hidden"],
            opt["dim_word"],
            n_layers=opt["num_layers"],
            input_dropout_p=opt["input_dropout_p"],
            rnn_cell=opt['decode_rnn_type'],
            rnn_dropout_p=opt["rnn_dropout_p"],
            bidirectional=opt["bidirectional"],
            w_dropout_p=opt["w_dropout_p"],
            chunk_size=opt["chunk_size"],
            extra_word_dim = opt["extra_word_dim"]
            )
        model = S2VTAttModel(encoder, decoder).cuda()
    #model = nn.DataParallel(model)
    # Setup the model
    model.load_state_dict(torch.load(opt["saved_model"]))
    crit = utils.LanguageModelCriterion()

    test(model, crit, dataset, dataset.get_vocab(), opt)


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
