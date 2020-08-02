import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from .Attention import Attention
from PLAIN_ON_LSTM import PLAIN_ONLSTMStack
import numpy as np


class DecoderSyntaxRNN(nn.Module):
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
                 parse_size,
                 max_len,
                 dim_hidden,
                 dim_parse,
                 n_layers=2,
                 rnn_cell='onlstm',
                 bidirectional=False,
                 input_dropout_p=0.1,
                 rnn_dropout_p=0.1,
                 w_dropout_p = 0.4):
    
        super(DecoderSyntaxRNN, self).__init__()

        self.bidirectional_encoder = bidirectional
        self.dim_output = parse_size
        self.dim_hidden = dim_hidden * 2 if bidirectional else dim_hidden
        self.dim_parse = dim_parse
        self.max_length = max_len
        self.sos_id = 1
        self.eos_id = 0
        self.input_dropout = nn.Dropout(input_dropout_p)
        self.attention = Attention(self.dim_hidden,self.dim_hidden,n_layers)

        self.embedding = nn.Embedding(self.dim_output, dim_parse)
        self.rnn_cell = PLAIN_ONLSTMStack


        if rnn_cell.lower() == 'onlstm':
            self.rnn = self.rnn_cell(
                [self.dim_hidden + dim_parse] + [self.dim_hidden] * n_layers,
                dropconnect=w_dropout_p,
                dropout=rnn_dropout_p
            )
        else:
            self.rnn = self.rnn_cell(
                self.dim_hidden + dim_parse,
                self.dim_hidden,
                n_layers,
                batch_first=True,
                dropout=rnn_dropout_p)

        self.out = nn.Linear(self.dim_hidden, self.dim_output)

        self._init_weights()

    def forward(self,
                encoder_outputs,
                encoder_hidden,
                targets=None,
                mode='train',
                opt={}):

        sample_max = opt.get('sample_max', 1)
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)

        batch_size, _, _ = encoder_outputs.size()

        seq_logprobs = []
        seq_preds = []

        decoder_hidden = self.init_hidden(batch_size)

        if mode == 'train':
            targets_emb = self.embedding(targets)
            for i in range(self.max_length - 1):
                current_words = targets_emb[:, i, :]
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

                decoder_input = torch.cat([current_words, context], dim=1)
                decoder_input = self.input_dropout(decoder_input).unsqueeze(1)
                decoder_input = decoder_input.permute(1,0,2)
                decoder_output, decoder_hidden, raw_outputs, outputs = self.rnn(decoder_input, decoder_hidden)
                decoder_output = decoder_output.permute(1,0,2)

                out_tmp = self.out(decoder_output.squeeze(1))
                logprobs = F.log_softmax(out_tmp, dim=1)
                seq_logprobs.append(logprobs.unsqueeze(1))

            seq_logprobs = torch.cat(seq_logprobs, 1)

        elif mode == 'inference':

            if beam_size > 1:
                prev_state = list(decoder_hidden)
                attention_hidden, attention_cell = prev_state[0]
                return self.sample_beam(encoder_outputs, attention_hidden, opt)
    

            for t in range(self.max_length - 1):
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

                if t == 0:  # input <bos>
                    it = torch.LongTensor([self.sos_id] * batch_size).cuda()
                elif sample_max:
                    sampleLogprobs, it = torch.max(logprobs, 1)
                    seq_logprobs.append(sampleLogprobs.view(-1, 1))
                    it = it.view(-1).long()
                else:
                    if temperature == 1.0:
                        prob_prev = torch.exp(logprobs)
                    else:
                        prob_prev = torch.exp(torch.div(logprobs, temperature))
                    it = torch.multinomial(prob_prev, 1).cuda()
                    sampleLogprobs = logprobs.gather(1, it)
                    seq_logprobs.append(sampleLogprobs.view(-1, 1))
                    it = it.view(-1).long()

                seq_preds.append(it.view(-1, 1))

                xt = self.embedding(it)
                decoder_input = torch.cat([xt, context], dim=1)
                decoder_input = self.input_dropout(decoder_input).unsqueeze(1)

                decoder_input = decoder_input.permute(1,0,2)
                decoder_output, decoder_hidden, raw_outputs, outputs = self.rnn(decoder_input, decoder_hidden)
                decoder_output = decoder_output.permute(1,0,2)
                    
                out_tmp = self.out(decoder_output.squeeze(1))
                logprobs = F.log_softmax(out_tmp, dim=1) #(batch_size, vocab_size)

            seq_logprobs = torch.cat(seq_logprobs, 1)
            seq_preds = torch.cat(seq_preds[1:], 1)

        return seq_logprobs, seq_preds

    def _init_weights(self):
        """ init the weight of some layers
        """
        nn.init.xavier_normal_(self.out.weight)

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