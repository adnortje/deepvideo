# imports
import torch
import torch.nn as nn

"""
Class ConvGRU_cell

    Creates a single Convolutional Gated Recurrent Unit cell

        Args:
            input_dim  (int)     : number of channels in input
            hidden_dim (int)     : dimension of cell's hidden state 
            kernel_i   (int)     : size of filter applied to input to cell
            stride_i   (int)     : stride applied to input convolution
            kernel_h   (int)     : size of filter applied to the previous hidden state
            stride_h   (int)     : stride applied to hidden state convolution
            padding_i  (int)     : padding applied around input
            padding_h  (int)     : padding applied to previous hidden state
            bias       (boolean) : True to include bias terms in convolution
"""


class ConvGruCell(nn.Module):

    def __init__(self,
                 input_dim,
                 hidden_dim,
                 kernel_i, stride_i, kernel_h, stride_h,
                 padding_i=0, dilation_i=1, groups_i=1,
                 padding_h=0, dilation_h=1, groups_h=1,
                 bias=False):

        super(ConvGruCell, self).__init__()

        self.hidden_dim = hidden_dim

        self.conv_in = nn.Conv2d(
            in_channels=input_dim,
            out_channels=3*hidden_dim,
            kernel_size=kernel_i,
            stride=stride_i,
            padding=padding_i,
            dilation=dilation_i,
            groups=groups_i,
            bias=bias
        )

        self.conv_h = nn.Conv2d(
            in_channels=hidden_dim,
            out_channels=3 * hidden_dim,
            kernel_size=kernel_h,
            stride=stride_h,
            padding=padding_h,
            dilation=dilation_h,
            groups=groups_h,
            bias=bias
        )

    def forward(self, x, h_prev):

        x = self.conv_in(x)
        h = self.conv_h(h_prev)
        x_z, x_r, x_n = torch.split(x, self.hidden_dim, dim=1)
        h_z, h_r, h_n = torch.split(h, self.hidden_dim, dim=1)

        # GRU logic
        z = torch.sigmoid(x_z + h_z)
        r = torch.sigmoid(x_r + h_r)
        n = torch.tanh(x_n + r * h_n)
        h_t = (1 - z) * n + z * h_prev

        return h_t

    def init_hidden(self, batch_size, h, w):
        if torch.cuda.is_available():
            h = torch.zeros(batch_size, self.hidden_dim, h, w).cuda()
        else:
            h = torch.zeros(batch_size, self.hidden_dim, h, w)
        return h