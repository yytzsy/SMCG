import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from .Attention_update import Attention_update
from PLAIN_ON_LSTM import PLAIN_ONLSTMStack
import numpy as np


class DecoderVideoRNN(nn.Module):
    """
    Provides functionality for decoding in a seq2seq framework, with an option for attention.
    Args:
        vocab_size (int): size of the vocabulary
        max_len (int): a maximum allowed length for the sequence to be processed
        dim_hidden (int): the number of features in the hidden state `h`
        n_layers (int, optional): number of recurrent layers (default: 1)
        rnn_cell (str, optional): type of RNN cell (default: gru)
        bidirectional (bool, optional): if the encoder is bidirectional (default False)
        input_dropout_p (float, optional): dropout probability for the input sequence (default: 0)
        rnn_dropout_p (float, optional): dropout probability for the output sequence (default: 0)

    """

    def __init__(self,
                 max_video_length,
                 dim_hidden,
                 dim_video,
                 n_layers=1,
                 bidirectional=False,
                 input_dropout_p=0.1,
                 rnn_dropout_p=0.1,
                 w_dropout_p = 0.4):
    
        super(DecoderVideoRNN, self).__init__()

        self.bidirectional_encoder = bidirectional
        self.dim_hidden = dim_hidden * 2 if bidirectional else dim_hidden
        self.max_video_length = max_video_length
        self.input_dropout = nn.Dropout(input_dropout_p)
        self.attention = Attention_update(dim_video, self.dim_hidden,n_layers)

        self.rnn_cell = PLAIN_ONLSTMStack
        self.rnn = self.rnn_cell(
            [self.dim_hidden] + [self.dim_hidden] * (n_layers-1)+ [dim_video],
            dropconnect=w_dropout_p,
            dropout=rnn_dropout_p
        )


    def forward(self,
                encoder_outputs,
                encoder_hidden,
                mode='train',
                opt={}):

        batch_size, _, _ = encoder_outputs.size()

        decoder_hidden = self.init_hidden(batch_size)

        decoder_output_list = []

        if mode == 'train':
            for i in range(self.max_video_length):
                prev_state = list(decoder_hidden)
                if len(prev_state)==1: # number of layers
                    attention_hidden, attention_cell = prev_state[0]
                else:
                    for jj in range(len(prev_state)):
                        hidden_item, cell_item = prev_state[jj]
                        if jj > 0:
                            attention_hidden = torch.cat((attention_hidden,hidden_item),-1)
                            attention_cell = torch.cat((attention_cell,cell_item),-1)
                        else:
                            attention_hidden = hidden_item
                            attention_cell = cell_item
                context = self.attention(attention_hidden, encoder_outputs)

                decoder_input = context
                decoder_input = self.input_dropout(decoder_input).unsqueeze(1)
                decoder_input = decoder_input.permute(1,0,2)
                decoder_output, decoder_hidden, raw_outputs, outputs = self.rnn(decoder_input, decoder_hidden)
                decoder_output = decoder_output.permute(1,0,2)

                decoder_output_list.append(decoder_output)


            decoder_output_list = torch.cat(decoder_output_list, 1)
            decoder_output_list = decoder_output_list.squeeze()
            
        return decoder_output_list


    def _init_rnn_state(self, encoder_hidden):
        """ Initialize the encoder hidden state. """
        if encoder_hidden is None:
            return None
        if isinstance(encoder_hidden, tuple):
            encoder_hidden = tuple(
                [self._cat_directions(h) for h in encoder_hidden])
        else:
            encoder_hidden = self._cat_directions(encoder_hidden)
        return encoder_hidden

    def _cat_directions(self, h):
        """ If the encoder is bidirectional, do the following transformation.
            (#directions * #layers, #batch, dim_hidden) -> (#layers, #batch, #directions * dim_hidden)
        """
        if self.bidirectional_encoder:
            h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        return h


    def init_hidden(self, bsz):
        return self.rnn.init_hidden(bsz)