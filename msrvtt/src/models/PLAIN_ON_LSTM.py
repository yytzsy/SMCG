import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np

from locked_dropout import LockedDropout


class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class LinearDropConnect(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, dropout=0.):
        super(LinearDropConnect, self).__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias
        )
        self.dropout = dropout

    def sample_mask(self):
        if self.dropout == 0.:
            self._weight = self.weight
        else:
            mask = self.weight.new_empty(
                self.weight.size(),
                dtype=torch.uint8
            )
            mask.bernoulli_(self.dropout)
            self._weight = self.weight.masked_fill(mask, 0.)

    def forward(self, input, sample_mask=False):
        if self.training:
            if sample_mask:
                self.sample_mask()
            return F.linear(input, self._weight, self.bias)
        else:
            return F.linear(input, self.weight * (1 - self.dropout),
                            self.bias)


def cumsoftmax(x, dim=-1):
    return torch.cumsum(F.softmax(x, dim=dim), dim=dim)


class PLAIN_ONLSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size, dropconnect=0.):
        super(PLAIN_ONLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.ih = nn.Sequential(
            nn.Linear(input_size, hidden_size*4, bias=True),
        )

        self.hh = LinearDropConnect(hidden_size, hidden_size*4, bias=True, dropout=dropconnect)


        self.drop_weight_modules = [self.hh]


    def modulation(self,x,gamma,beta):
        eps = 1e-5
        x_mean = torch.mean(x, dim=-1, keepdim=True) 
        x_var = torch.mean((x - x_mean) ** 2, dim=-1, keepdim=True)
        x_hat = (x - x_mean) / torch.sqrt(x_var + eps)
        return gamma * x_hat + beta


    def forward(self, input, hidden, transformed_input=None):
        hx, cx = hidden

        if transformed_input is None:
            transformed_input = self.ih(input) # transform input W*x
        transformed_hidden = self.hh(hx) # self.hh(hx): transform hidden state U*h

        gates = transformed_input + transformed_hidden
        outgate, cell, ingate, forgetgate = gates.chunk(4,1) #original LSTM state(batch_size,hidden_size*4) => chunkprocess 4*(batch_size,hidden_size)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cell = torch.tanh(cell)
        outgate = torch.sigmoid(outgate)

        cy = forgetgate * cx + ingate * cell
        hy = outgate * torch.tanh(cy)

        return hy.view(-1, self.hidden_size), cy


    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (weight.new(bsz, self.hidden_size).zero_(),
                weight.new(bsz, self.hidden_size).zero_())


    def sample_masks(self):
        for m in self.drop_weight_modules:
            m.sample_mask()


class PLAIN_ONLSTMStack(nn.Module):
    def __init__(self, layer_sizes, dropout=0., dropconnect=0.):
        super(PLAIN_ONLSTMStack, self).__init__()
        self.cells = nn.ModuleList([PLAIN_ONLSTMCell(layer_sizes[i],
                                               layer_sizes[i+1],
                                               dropconnect=dropconnect)
                                    for i in range(len(layer_sizes) - 1)])
        self.lockdrop = LockedDropout()
        self.dropout = dropout
        self.sizes = layer_sizes

    def init_hidden(self, bsz):
        return [c.init_hidden(bsz) for c in self.cells]

    def forward(self, input, hidden):
        length, batch_size, _ = input.size()

        if self.training:
            for c in self.cells:
                c.sample_masks()

        prev_state = list(hidden)
        prev_layer = input
        raw_outputs = []
        outputs = []
        
        for l in range(len(self.cells)):
            curr_layer = [None] * length
            t_input = self.cells[l].ih(prev_layer)

            for t in range(length):
                hidden, cell = self.cells[l](
                    None, prev_state[l], transformed_input=t_input[t]
                )
                prev_state[l] = hidden, cell  # overwritten every timestep
                curr_layer[t] = hidden

            prev_layer = torch.stack(curr_layer)
            raw_outputs.append(prev_layer)
            if l < len(self.cells) - 1:
                prev_layer = self.lockdrop(prev_layer, self.dropout)
            outputs.append(prev_layer)
        output = prev_layer

        return output, prev_state, raw_outputs, outputs


