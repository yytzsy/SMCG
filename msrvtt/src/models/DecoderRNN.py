import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from .Attention import Attention
from ON_LSTM import ONLSTMStack
import numpy as np

class DecoderRNN(nn.Module):
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
                 vocab_size,
                 max_len,
                 dim_hidden,
                 dim_word,
                 dim_parse,
                 n_layers=2,
                 rnn_cell='onlstm',
                 bidirectional=False,
                 input_dropout_p=0.1,
                 rnn_dropout_p=0.1,
                 w_dropout_p = 0.4,
                 embedding_pretrained_weights = None):
        super(DecoderRNN, self).__init__()

        self.bidirectional_encoder = bidirectional
        self.dim_output = vocab_size
        self.dim_hidden = dim_hidden * 2 if bidirectional else dim_hidden
        self.dim_word = dim_word
        self.dim_parse = dim_parse
        self.max_length = max_len
        self.sos_id = 1
        self.eos_id = 0
        self.input_dropout_input = nn.Dropout(input_dropout_p)
        self.input_dropout_control = nn.Dropout(input_dropout_p)

        self.attention_video = Attention(self.dim_hidden,self.dim_hidden,n_layers)
        self.attention_parse = Attention(self.dim_hidden,self.dim_parse,n_layers)

        self.embedding = nn.Embedding(self.dim_output, dim_word)
        if embedding_pretrained_weights is not None:
            self.embedding.weight = nn.Parameter(torch.FloatTensor(embedding_pretrained_weights))


        self.rnn_type = rnn_cell

        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU
        elif rnn_cell.lower() == 'onlstm':
            self.rnn_cell = ONLSTMStack


        if rnn_cell.lower() == 'onlstm':
            self.rnn = self.rnn_cell(
                [self.dim_hidden + dim_word] + [self.dim_hidden] * n_layers,
                dim_parse,
                dropconnect=w_dropout_p,
                dropout=rnn_dropout_p
            )
        else:
            self.rnn = self.rnn_cell(
                2*self.dim_hidden + dim_word,
                self.dim_hidden,
                n_layers,
                batch_first=True,
                dropout=rnn_dropout_p)

        self.out = nn.Linear(self.dim_hidden, self.dim_output)

        self._init_weights()

    def forward(self,
                encoder_outputs,
                encoder_hidden,
                encoder_outputs_parse,
                encoder_hidden_parse,
                targets=None,
                mode='train',
                opt={}):
        """

        Inputs: inputs, encoder_hidden, encoder_outputs, function, teacher_forcing_ratio
        - **encoder_hidden** (num_layers * num_directions, batch_size, dim_hidden): tensor containing the features in the
          hidden state `h` of encoder. Used as the initial hidden state of the decoder. (default `None`)
        - **encoder_outputs** (batch, seq_len, dim_hidden * num_directions): (default is `None`).
        - **targets** (batch, max_length): targets labels of the ground truth sentences

        Outputs: seq_probs,
        - **seq_logprobs** (batch_size, max_length, vocab_size): tensors containing the outputs of the decoding function.
        - **seq_preds** (batch_size, max_length): predicted symbols
        """


        sample_max = opt.get('sample_max', 1)
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)

        batch_size, _, _ = encoder_outputs.size()

        seq_logprobs = []
        seq_preds = []
        if self.rnn_type != 'onlstm':
            self.rnn.flatten_parameters()
        if self.rnn_type == 'onlstm':
            decoder_hidden = self.init_hidden(batch_size)
        else:
            decoder_hidden = self._init_rnn_state(encoder_hidden)

        word_feats_list = []

        if mode == 'train':
            # use targets as rnn inputs
            targets_emb = self.embedding(targets)
            for i in range(self.max_length - 1):
                current_words = targets_emb[:, i, :]
                if self.rnn_type == 'onlstm':
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
                    context_video = self.attention_video(attention_hidden, encoder_outputs)
                    context_parse = self.attention_parse(attention_hidden, encoder_outputs_parse)
                else:
                    context_video = self.attention_video(decoder_hidden.squeeze(0), encoder_outputs)
                    context_parse = self.attention_parse(decoder_hidden.squeeze(0), encoder_outputs_parse)

                decoder_input = torch.cat([current_words, context_video], dim=1)
                decoder_input = self.input_dropout_input(decoder_input).unsqueeze(1)
                control_input = self.input_dropout_control(context_parse).unsqueeze(1)
                if self.rnn_type != 'onlstm':
                    decoder_output, decoder_hidden = self.rnn(decoder_input, decoder_hidden)
                else:
                    decoder_input = decoder_input.permute(1,0,2)
                    control_input = control_input.permute(1,0,2)
                    decoder_output, decoder_hidden, raw_outputs, outputs = self.rnn(decoder_input, decoder_hidden, control_input)
                    decoder_output = decoder_output.permute(1,0,2)


                out_tmp = self.out(decoder_output.squeeze(1))
                logprobs = F.log_softmax(out_tmp, dim=1)
                seq_logprobs.append(logprobs.unsqueeze(1))

                word_feats_list.append(decoder_output)

            seq_logprobs = torch.cat(seq_logprobs, 1)
            word_feats_list = torch.stack(word_feats_list,0)
            word_feats_list = word_feats_list.squeeze()


        elif mode == 'inference':

            if beam_size > 1:
                if self.rnn_type == 'onlstm':
                    prev_state = list(decoder_hidden)
                    attention_hidden, attention_cell = prev_state[0]
                    return self.sample_beam(encoder_outputs, attention_hidden, opt)
                else:
                    return self.sample_beam(encoder_outputs, decoder_hidden, opt)

            for t in range(self.max_length - 1):
                if self.rnn_type == 'onlstm':
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
                    context_video = self.attention_video(attention_hidden, encoder_outputs)
                    context_parse = self.attention_parse(attention_hidden, encoder_outputs_parse)
                else:
                    context_video = self.attention_video(decoder_hidden.squeeze(0), encoder_outputs)
                    context_parse = self.attention_parse(decoder_hidden.squeeze(0), encoder_outputs_parse)
                
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
                decoder_input = torch.cat([xt, context_video], dim=1)
                decoder_input = self.input_dropout_input(decoder_input).unsqueeze(1)
                # control_input = torch.cat([context_parse, context_video], dim=1)
                control_input = self.input_dropout_control(context_parse).unsqueeze(1)
                if self.rnn_type != 'onlstm':
                    decoder_output, decoder_hidden = self.rnn(decoder_input, decoder_hidden)
                else:
                    decoder_input = decoder_input.permute(1,0,2)
                    control_input = control_input.permute(1,0,2)
                    decoder_output, decoder_hidden, raw_outputs, outputs = self.rnn(decoder_input, decoder_hidden, control_input)
                    decoder_output = decoder_output.permute(1,0,2)
                    

                out_tmp = self.out(decoder_output.squeeze(1))
                logprobs = F.log_softmax(out_tmp, dim=1) #(batch_size, vocab_size)

                word_feats_list.append(decoder_output)

            seq_logprobs = torch.cat(seq_logprobs, 1)
            seq_preds = torch.cat(seq_preds[1:], 1)
            word_feats_list = torch.stack(word_feats_list,0)
            word_feats_list = word_feats_list.squeeze()

        return seq_logprobs, seq_preds, word_feats_list

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