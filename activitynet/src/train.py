import json
import os

import numpy as np

import misc.utils as utils
import opts
import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_value_
from dataloader import VideoDataset
from misc.rewards import get_self_critical_reward, init_cider_scorer
from models import DecoderRNN, EncoderRNN, S2VTAttModel, S2VTModel, DecoderSyntaxRNN, DecoderVideoRNN
from torch import nn
from torch.utils.data import DataLoader


import sys
reload(sys)
sys.setdefaultencoding('utf-8')

def train(loader, model, crit, optimizer, lr_scheduler, opt, rl_crit=None, start_epoch=0):
    model.train()
    #model = nn.DataParallel(model)
    for epoch in range(opt["epochs"]):
        lr_scheduler.step()

        iteration = 0
        # If start self crit training
        if opt["self_crit_after"] != -1 and epoch >= opt["self_crit_after"]:
            sc_flag = True
            init_cider_scorer(opt["cached_tokens"])
        else:
            sc_flag = False

        for data in loader:
            print os.path.abspath('./').split('/')[-1]
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            fc_feats = data['fc_feats'].cuda() # video features
            # random_fc_feat = data['random_fc_feat'].cuda()
            labels = data['labels'].cuda()
            masks = data['masks'].cuda()
            parse_labels = data['label_parse'].cuda()
            parse_masks = data['mask_parse'].cuda()
            mask_fc_feat = data['mask_fc_feat'].cuda()

            optimizer.zero_grad()

            seq_probs, _, seq_prob_syntax, _, reconstruct_video_sequence = model(fc_feats, labels, parse_labels, 'train', opt)
            loss, loss_syntax, loss_content, loss_caption = crit(seq_probs, labels[:, 1:], masks[:, 1:], seq_prob_syntax, parse_labels[:,1:], parse_masks[:,1:], reconstruct_video_sequence, fc_feats, mask_fc_feat)

            loss.backward()
            clip_grad_value_(model.parameters(), opt['grad_clip'])
            optimizer.step()

            train_loss = loss.item()
            train_loss_syntax = loss_syntax.item()
            train_loss_content = loss_content.item()
            train_loss_caption = loss_caption.item()

            torch.cuda.synchronize()
            iteration += 1

            if not sc_flag:
                print("iter %d (epoch %d), train_loss = %.6f, train_loss_syntax = %.6f, train_loss_content = %.6f, train_loss_caption = %.6f" %
                      (iteration, epoch+start_epoch, train_loss, train_loss_syntax, train_loss_content, train_loss_caption))
            else:
                print("iter %d (epoch %d), avg_reward = %.6f" %
                      (iteration, epoch+start_epoch, np.mean(reward[:, 0])))

        current_epoch = epoch+start_epoch
        if current_epoch % opt["save_checkpoint_every"] == 0:
            model_path = os.path.join(opt["checkpoint_path"],
                                      'model_%d.pth' % (current_epoch))
            model_info_path = os.path.join(opt["checkpoint_path"],
                                           'model_score.txt')
            torch.save(model.state_dict(), model_path)
            print("model saved to %s" % (model_path))
            with open(model_info_path, 'a') as f:
                f.write("model_%d, loss: %.6f\t" % (current_epoch, train_loss))
                f.write("train_loss_syntax: %.6f\t" % (train_loss_syntax))
                f.write("train_loss_content: %.6f\t" % (train_loss_content))
                f.write("train_loss_caption: %.6f\n" % (train_loss_caption))


def main(opt):
    dataset = VideoDataset(opt, 'train')
    dataloader = DataLoader(dataset, batch_size=opt["batch_size"], shuffle=True)
    opt["vocab_size"] = dataset.get_vocab_size()
    opt["parse_size"] = dataset.get_parse_vocab_size()


    if opt["model"] == 'S2VTModel':
        model = S2VTModel(
            opt["vocab_size"],
            opt["max_len"],
            opt["dim_hidden"],
            opt["dim_word"],
            opt['dim_vid'],
            rnn_cell=opt['rnn_type'],
            n_layers=opt['num_layers'],
            rnn_dropout_p=opt["rnn_dropout_p"])
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
        model = S2VTAttModel(encoder, decoder, decoder_syntax, decoder_video, opt['dim_word'], opt['dim_hidden'])
        print model
    model = model.cuda()
    crit = utils.LanguageModelCriterion_withSyntax(opt['caption_alpha'],opt['syntax_alpha'],opt['content_alpha'])
    optimizer = optim.Adam(
        model.parameters(),
        lr=opt["learning_rate"],
        weight_decay=opt["weight_decay"])
    exp_lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=opt["learning_rate_decay_every"],
        gamma=opt["learning_rate_decay_rate"])

    start_epoch = 0
    # pretrained_model = './save/model_'+str(start_epoch-1)+'.pth'
    # model.load_state_dict(torch.load(pretrained_model))
    train(dataloader, model, crit, optimizer, exp_lr_scheduler, opt, start_epoch = start_epoch)


if __name__ == '__main__':
    opt = opts.parse_opt()
    opt = vars(opt)
    os.environ['CUDA_VISIBLE_DEVICES'] = opt["gpu"]
    opt_json = os.path.join(opt["checkpoint_path"], 'opt_info.json')
    if not os.path.isdir(opt["checkpoint_path"]):
        os.mkdir(opt["checkpoint_path"])
    with open(opt_json, 'w') as f:
        json.dump(opt, f)
    print('save opt details to %s' % (opt_json))
    main(opt)
