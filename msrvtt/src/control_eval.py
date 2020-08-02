import json
import os
import argparse
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from models import EncoderRNN, DecoderRNN, S2VTAttModel, S2VTModel,DecoderSyntaxRNN, DecoderVideoRNN
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





def test(model, crit, dataset, vocab, opt, epoch):

    info_extend = json.load(open(opt["info_json_extend"]))
    ix_to_word = info_extend['ix_to_word']

    model.eval()
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    results = []
    output_dict = {}

    if not os.path.exists(opt["results_path"]):
        os.makedirs(opt["results_path"])

    template_sentence_path = os.path.join(opt["results_path"], "control_template_"+str(epoch)+".txt")
    template_sentence_file = open(template_sentence_path, 'w')
    predict_sentence_path = os.path.join(opt["results_path"], "control_predict_"+str(epoch)+".txt")
    predict_sentence_file = open(predict_sentence_path, 'w')
    groundtruth_sentence_path = os.path.join(opt["results_path"], "control_ground_"+str(epoch)+".txt")
    groundtruth_sentence_file = open(groundtruth_sentence_path, 'w')

    sample_num = 0
    for data in loader:


        print sample_num
        sample_num +=1
        print "control_eval "+os.path.abspath('./').split('/')[-1]

        # forward the model to get loss
        video_ids = data['video_ids']
        fc_feats = data['fc_feats'].cuda() #(b,40,dim) b=1
        # random_fc_feat = data['random_fc_feat'].cuda()
        gts_labels = data['gts'].cuda() #(b,20,28)
        labels = data['labels'].cuda() #(b,28) b=1
        template_labels = data['template_labels'].cuda() #(b,20,28)
        template_parse_labels = data['template_parse_labels'].cuda() #(b,20,120) b=1


        groundtruth_sentence_list = []
        gts_labels_data = gts_labels.cpu().numpy()
        caption_gts_num = np.shape(gts_labels_data)[1]
        jj = 0
        while jj < caption_gts_num:
            sentence_label = gts_labels_data[0,jj,:]
            content = ''
            for iid in sentence_label:
                content = content + ' ' + ix_to_word[str(iid)]
                if ix_to_word[str(iid)] == '<eos>':
                    break
            content = content.strip()
            groundtruth_sentence_list.append(content)
            jj+=1


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
        # random_fc_feat = np.tile(random_fc_feat.cpu().numpy(),[template_gts_num,1,1])
        labels = np.tile(labels.cpu().numpy(),[template_gts_num,1])
        template_parse_labels = np.squeeze(template_parse_labels.cpu().numpy())

        fc_feats = torch.from_numpy(fc_feats).cuda()
        # random_fc_feat = torch.from_numpy(random_fc_feat).cuda()
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
            groundtruth_caption = groundtruth_sentence_list[min(k,caption_gts_num-1)]
            template_gts_caption = template_sentence_list[k]

            template_gts_caption = template_gts_caption.replace('<eos>','')
            template_gts_caption = template_gts_caption.replace('<sos>','')
            template_gts_caption = template_gts_caption.strip()
            sent = sent.replace('<eos>','')
            sent = sent.replace('<sos>','')
            sent = sent.strip()
            groundtruth_caption = groundtruth_caption.replace('<eos>','')
            groundtruth_caption = groundtruth_caption.replace('<sos>','')
            groundtruth_caption = groundtruth_caption.strip()
            
            template_sentence_file.write(template_gts_caption.strip()+'\n')
            predict_sentence_file.write(sent.strip()+'\n')
            groundtruth_sentence_file.write(groundtruth_caption.strip()+'\n')

            if video_id not in output_dict:
                output_dict[video_id] = {'predict':[],'template':[],'content':groundtruth_sentence_list}
            output_dict[video_id]['predict'].append(sent)
            output_dict[video_id]['template'].append(template_gts_caption)

    template_sentence_file.close()
    predict_sentence_file.close()
    groundtruth_sentence_file.close()
    pkl.dump(output_dict, open(os.path.join(opt["results_path"],'control_output_dict'+str(epoch)+'.pkl'),'w'))

    ################################### calculate syntactic distance (tree edit distance) ############################

    f = open(template_sentence_path,'r')
    all_output_template_sentence = f.readlines()
    f.close()
    f = open(predict_sentence_path,'r')
    all_output_predict_sentence = f.readlines()
    f.close()

    template_path_list = []
    predict_path_list = []
    file_count = 0
    tmp_template_sentence_path = os.path.join(opt["results_path"], "control_template_"+str(epoch)+"_"+str(file_count)+".txt")
    tmp_template_sentence_file = open(tmp_template_sentence_path,'w')
    tmp_predict_sentence_path = os.path.join(opt["results_path"], "control_predict_"+str(epoch)+"_"+str(file_count)+".txt")
    tmp_predict_sentence_file = open(tmp_predict_sentence_path,'w')
    for kk in range(len(all_output_predict_sentence)):
        if kk > 0 and kk % 5000 == 0:
            tmp_template_sentence_file.close()
            tmp_predict_sentence_file.close()
            template_path_list.append(tmp_template_sentence_path)
            predict_path_list.append(tmp_predict_sentence_path)
            file_count+=1
            tmp_template_sentence_path = os.path.join(opt["results_path"], "control_template_"+str(epoch)+"_"+str(file_count)+".txt")
            tmp_template_sentence_file = open(tmp_template_sentence_path,'w')
            tmp_predict_sentence_path = os.path.join(opt["results_path"], "control_predict_"+str(epoch)+"_"+str(file_count)+".txt")
            tmp_predict_sentence_file = open(tmp_predict_sentence_path,'w')
        tmp_template_sentence_file.write(all_output_template_sentence[kk])
        tmp_predict_sentence_file.write(all_output_predict_sentence[kk])

    if tmp_template_sentence_path not in template_path_list:
        template_path_list.append(tmp_template_sentence_path)
    if tmp_predict_sentence_path not in predict_path_list:
        predict_path_list.append(tmp_predict_sentence_path)
    ##############################################################################################################################
    spe = stanford_parsetree_extractor()
    ref_parses = spe.run(template_path_list)
    pre_parses = spe.run(predict_path_list)
    all_ted = []
    for ref, pre in zip(ref_parses, pre_parses):
        ted = compute_tree_edit_distance(ref, pre)
        all_ted.append(ted)
    mean_ted = np.mean(np.array(all_ted))
    mean_ted_dict = {'mean_ted':mean_ted}

    for f in template_path_list:
        os.remove(f)
    for f in predict_path_list:
        os.remove(f)
    #############################################################################################################################


    print 'mean_ted = '+str(mean_ted)
    with open(os.path.join(opt["results_path"], "control_scores.txt"), 'a') as scores_table:
        scores_table.write('Epoch: '+str(epoch))
        scores_table.write(json.dumps(mean_ted_dict)+ "\n")


def main(opt):


    # sub_dir =  'result_epoch_'+opt["saved_model"].split('_')[1].split('.')[0]
    # time_stamp = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
    # logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
    # log_file_name = str(time_stamp)+'_'+sub_dir+'.log'
    # fh = logging.FileHandler(filename=log_file_name, mode='w', encoding='utf-8')
    # fh.setFormatter(logging.Formatter('%(message)s'))
    # fh.setLevel(logging.INFO)
    # logging.root.addHandler(fh)

    dataset = VideoDataset(opt, "test")
    opt["vocab_size"] = dataset.get_vocab_size()
    opt["parse_size"] = dataset.get_parse_vocab_size()


    # parse_embedding = np.load(opt['parse_embedding'])
    # decoder_embedding = np.load(opt['decoder_embedding'])


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
