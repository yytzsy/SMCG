import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Attention_update(nn.Module):
    """
    Applies an attention mechanism on the output features from the decoder.
    """

    def __init__(self, dim_key, dim, num_layers):
        super(Attention_update, self).__init__()
        self.dim = dim
        self.dim_key = dim_key
        self.num_layers = num_layers
        self.linear1 = nn.Linear(dim_key + dim * self.num_layers, dim)
        self.linear2 = nn.Linear(dim, 1, bias=False)
        #self._init_hidden()

    def _init_hidden(self):
        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.xavier_normal_(self.linear2.weight)

    def forward(self, hidden_state, encoder_outputs):
        """
        Arguments:
            hidden_state {Variable} -- batch_size x dim
            encoder_outputs {Variable} -- batch_size x seq_len x dim

        Returns:
            Variable -- context vector of size batch_size x dim
        """
        batch_size, seq_len, _ = encoder_outputs.size()
        hidden_state = hidden_state.unsqueeze(1).repeat(1, seq_len, 1)
        inputs = torch.cat((encoder_outputs, hidden_state),2).view(-1, self.dim_key + self.dim * self.num_layers)
        o = self.linear2(torch.tanh(self.linear1(inputs)))
        e = o.view(batch_size, seq_len)
        alpha = F.softmax(e, dim=1)
        context = torch.bmm(alpha.unsqueeze(1), encoder_outputs).squeeze(1)
        return context
