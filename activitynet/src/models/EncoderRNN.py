import torch.nn as nn
import torch

class EncoderRNN(nn.Module):
    def __init__(self, dim_parse, parse_num, dim_vid, dim_hidden, input_dropout_p=0.2, rnn_dropout_p=0.5,
                 n_layers=1, bidirectional=False, rnn_cell='gru', embedding_pretrained_weights = None):
        """

        Args:
            hidden_dim (int): dim of hidden state of rnn
            input_dropout_p (int): dropout probability for the input sequence
            dropout_p (float): dropout probability for the output sequence
            n_layers (int): number of rnn layers
            rnn_cell (str): type of RNN cell ('LSTM'/'GRU')
        """
        super(EncoderRNN, self).__init__()
        self.dim_parse = dim_parse
        self.parse_num = parse_num
        self.dim_vid = dim_vid
        self.dim_hidden = dim_hidden
        self.input_dropout_p = input_dropout_p
        self.rnn_dropout_p = rnn_dropout_p
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.rnn_cell = rnn_cell

        self.vid2hid = nn.Linear(dim_vid, dim_hidden)
        self.parse2hid = nn.Linear(dim_parse, dim_parse)
        self.input_dropout_video = nn.Dropout(input_dropout_p)
        self.input_dropout_parse = nn.Dropout(input_dropout_p)

        self.parseembedding = nn.Embedding(parse_num, dim_parse)
        if embedding_pretrained_weights is not None:
            self.parseembedding.weight = nn.Parameter(torch.FloatTensor(embedding_pretrained_weights))

        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU

        self.rnn_video = self.rnn_cell(input_size = dim_hidden, hidden_size = dim_hidden, num_layers = n_layers, batch_first=True,
                                bidirectional=bool(bidirectional), dropout=self.rnn_dropout_p)


        self.rnn_parse = self.rnn_cell(input_size = dim_parse, hidden_size = dim_parse, num_layers = n_layers, batch_first=True,
                                bidirectional=bool(bidirectional), dropout=self.rnn_dropout_p)
        self._init_hidden()

    def _init_hidden(self):
        nn.init.xavier_normal_(self.vid2hid.weight)
        nn.init.xavier_normal_(self.parse2hid.weight)

    def forward(self, vid_feats, labels_parse):
        """
        Applies a multi-layer RNN to an input sequence.
        Args:
            input_var (batch, seq_len): tensor containing the features of the input sequence.
            input_lengths (list of int, optional): A list that contains the lengths of sequences
              in the mini-batch
        Returns: output, hidden
            - **output** (batch, seq_len, hidden_size): variable containing the encoded features of the input sequence
            - **hidden** (num_layers * num_directions, batch, hidden_size): variable containing the features in the hidden state h
        """
        batch_size, seq_len, dim_vid = vid_feats.size()
        vid_feats = self.vid2hid(vid_feats.view(-1, dim_vid))
        vid_feats = self.input_dropout_video(vid_feats)
        vid_feats = vid_feats.view(batch_size, seq_len, self.dim_hidden)
        self.rnn_video.flatten_parameters()
        output, hidden = self.rnn_video(vid_feats)

        # batch_size, seq_len, dim_vid = random_vid_feats.size()
        # random_vid_feats = self.vid2hid(random_vid_feats.view(-1, dim_vid))
        # random_vid_feats = self.input_dropout_video(random_vid_feats)
        # random_vid_feats = random_vid_feats.view(batch_size, seq_len, self.dim_hidden)
        # self.rnn_video.flatten_parameters()
        # output_random, hidden_random = self.rnn_video(random_vid_feats)

        parse_feats = self.parseembedding(labels_parse)
        batch_size, seq_len, dim_parse = parse_feats.size()
        parse_feats = self.parse2hid(parse_feats.view(-1, dim_parse))
        parse_feats = self.input_dropout_parse(parse_feats)
        parse_feats = parse_feats.view(batch_size, seq_len, dim_parse)
        self.rnn_parse.flatten_parameters()
        parse_output, parse_hidden = self.rnn_parse(parse_feats)

        return output, hidden, parse_output, parse_hidden
