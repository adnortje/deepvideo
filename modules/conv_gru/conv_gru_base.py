import torch
import numbers
import warnings
import torch.nn as nn
from .conv_gru_cell import ConvGruCell


class ConvGRU(nn.Module):

    def __init__(self,
                 input_dim,
                 hidden_dim,
                 kernel_i, stride_i, padding_i, dilation_i, groups_i,
                 kernel_h, stride_h, padding_h, dilation_h, groups_h,
                 num_layers=1,
                 bias=False,
                 batch_first=True,
                 dropout=0,
                 bidirectional=False):

        super(ConvGRU, self).__init__()

        # rnn parameters
        self.bias = bias
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # sanity check dropout value
        self.dropout = dropout
        if not isinstance(dropout, numbers.Number) or not 0 <= dropout <= 1 or isinstance(dropout, bool):
            raise ValueError("Dropout should be a number in range [0, 1]!")
        if dropout > 0 and num_layers == 1:
            warnings.warn("dropout option adds dropout after all but last "
                          "recurrent layer, so non-zero dropout expects "
                          "num_layers greater than 1, but got dropout={} and "
                          "num_layers={}".format(dropout, num_layers))

        # def number of directions
        self.bidirectional = bidirectional

        # def Convolutional GRU cells
        self.gru_cells_forward = nn.ModuleList()

        if bidirectional:
            self.gru_cells_reverse = nn.ModuleList()

        for n in range(self.num_layers):

            if n == 0:
                input_dim = self.input_dim
            else:
                input_dim = self.hidden_dim[n-1]

            # append cell
            self.gru_cells_forward.append(
                ConvGruCell(
                    input_dim,
                    self.hidden_dim[n],
                    *(kernel_i[n], stride_i[n], padding_i[n], dilation_i[n], groups_i[n]),
                    *(kernel_h[n], stride_h[n], padding_h[n], dilation_h[n], groups_h[n]),
                    bias=bias
                )
            )

            # if bidirectional
            if self.bidirectional:

                # append reverse cell
                self.gru_cells_reverse.append(
                    ConvGruCell(
                        input_dim,
                        self.hidden_dim[n],
                        *(kernel_i[n], stride_i[n], padding_i[n], dilation_i[n], groups_i[n]),
                        *(kernel_h[n], stride_h[n], padding_h[n], dilation_h[n], groups_h[n]),
                        bias=bias
                    )
                )

    def forward(self, x, init_states=None):

        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            x = input.permute(1, 0, 2, 3, 4)

        if init_states is None:
            # initialize s0 states
            init_states = self._init_hidden(x.size(0))

        # def seq length
        t_seq = x.size(1)

        f_seq = []
        r_seq = []
        output_seq =[]

        for n in range(self.num_layers):

            # init ht for t0
            ht = init_states[n]
            r_ht = init_states[n]

            for t in range(t_seq):

                ht = self.gru_cells_forward[n](x[:, t, :, :, :], ht)
                f_seq.append(ht)

                if self.bidirectional:
                    r_ht = self.gru_cells_reverse[n](x[:, (t_seq - t - 1), :, :, :], r_ht)
                    r_seq.append(r_ht)

            # concat reverse and forward outputs
            x = torch.cat((f_seq, r_seq), dim=0)

            x = torch.cat(output_seq, dim=1)

            if n < self.num_layers - 1 and self.dropout != 0:
                # add dropout to all but last layer
                x = nn.Dropout(p=self.dropout)

        return x

    def _init_hidden(self, batch_size):
        init_states = []
        for n in range(self.num_layers):
            init_states.append(self.gru_cell_list[n].init_hidden(batch_size))

        return init_states





