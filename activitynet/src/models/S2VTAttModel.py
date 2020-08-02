import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F

class S2VTAttModel(nn.Module):
    def __init__(self, encoder, decoder, decoder_syntax, decoder_video, dim_word, dim_hidden):
        """

        Args:
            encoder (nn.Module): Encoder rnn
            decoder (nn.Module): Decoder rnn
        """
        super(S2VTAttModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.decoder_syntax = decoder_syntax
        self.decoder_video = decoder_video

        self.dim_word = dim_word
        self.dim_hidden = dim_hidden


    def forward(self, vid_feats, target_variable, target_variable_parse, mode='train', opt={}):
        """

        Args:
            vid_feats (Variable): video feats of shape [batch_size, seq_len, dim_vid]
            target_variable (None, optional): groung truth labels

        Returns:
            seq_prob: Variable of shape [batch_size, max_len-1, vocab_size]
            seq_preds: [] or Variable of shape [batch_size, max_len-1]
        """
        encoder_outputs, encoder_hidden, encoder_outputs_parse, encoder_hidden_parse = self.encoder(vid_feats, target_variable_parse)
        seq_prob, seq_preds, decoder_word_hidden_sequence = self.decoder(encoder_outputs, encoder_hidden, encoder_outputs_parse, encoder_hidden_parse, target_variable, mode, opt)
        
        decoder_word_hidden_sequence = decoder_word_hidden_sequence.permute(1,0,2)
            
        seq_prob_syntax, seq_preds_syntax = self.decoder_syntax(encoder_outputs = decoder_word_hidden_sequence, encoder_hidden = None, targets = target_variable_parse, mode = mode, opt = opt)
        reconstruct_video_sequence = self.decoder_video(encoder_outputs = decoder_word_hidden_sequence, encoder_hidden = None, mode = mode, opt = opt)
        
        return seq_prob, seq_preds, seq_prob_syntax, seq_preds_syntax, reconstruct_video_sequence
