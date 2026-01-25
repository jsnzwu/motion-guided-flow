import torch.nn as nn
import torch
from utils.log import log
from wickit.utils.basic.string import dict_to_string
class ConvGRUCellV6(nn.Module):
    def __init__(self, hidden_channel, out_channel, kernel_size=1, bias=True):
        """
        Initialize the ConvLSTM cell
        :param input_size: (int, int)
            Height and width of input tensor as (height, width).
        :param input_dim: int
            Number of channels of input tensor.
        :param hidden_dim: int
            Number of channels of hidden state.
        :param kernel_size: (int, int)
            Size of the convolutional kernel.
        :param bias: bool
            Whether or not to add the bias.
        :param dtype: torch.cuda.FloatTensor or torch.FloatTensor
            Whether or not to use cuda.
        """
        super().__init__()
        self.padding = kernel_size // 2, kernel_size // 2
        self.hidden_channel = hidden_channel
        self.out_channel = out_channel
        self.bias = bias

        self.conv_gates = nn.Conv2d(in_channels=hidden_channel + out_channel,
                                    out_channels=out_channel * 2,  # for update_gate,reset_gate respectively
                                    kernel_size=kernel_size,
                                    padding=self.padding,
                                    bias=self.bias)

        self.conv_can = nn.Conv2d(in_channels=hidden_channel + out_channel,
                              out_channels=out_channel, # for candidate neural memory
                              kernel_size=kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, h_cur):
        assert isinstance(input_tensor, list)
        combined = torch.cat([*input_tensor, h_cur], dim=1)
        combined_conv = self.conv_gates(combined)

        gamma, beta = torch.split(combined_conv, self.out_channel, dim=1)
        reset_gate = torch.sigmoid(gamma)
        update_gate = torch.sigmoid(beta)

        combined = torch.cat([*input_tensor, reset_gate*h_cur], dim=1)
        cc_cnm = self.conv_can(combined)
        cnm = torch.tanh(cc_cnm)

        h_next = (1 - update_gate) * h_cur + update_gate * cnm
        return h_next

class ConvLSTMCellHiddenV6(nn.Module):

    def __init__(self, hidden_channel, out_channel, kernel_size=1, bias=True):
        super().__init__()
        self.hidden_channel = hidden_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias
        self.conv = nn.Conv2d(in_channels=self.hidden_channel,
                              out_channels=self.out_channel * 4,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)
        self.default_c_list = [self.out_channel] * 4

    def forward(self, input_tensor, cur_state, c_list=None):
        ''' 
        cc_i + cc_f + cc_g <--> input_tensor + cur_state
        cc_f <--> c_cur <--> cc_i <--> cc_g <--> c_next
        c_cur: long-term memory, last feature
        h_cur: short-term memory, g_buffer
        input_tensor: current feature(input)
        c_next: current feature(output)
        '''
        # log.debug(dict_to_string(input_tensor))
        combined = torch.cat(input_tensor, dim=1)  # concatenate along channel axis
        combined_conv = self.conv(combined)
        if c_list is None:
            c_list = self.default_c_list
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, c_list, dim=1) # type: ignore 
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * cur_state + i * g
        h_next = o * torch.tanh(c_next)

        return c_next, h_next
    
class ConvLSTMCellV6(nn.Module):
    def __init__(self, hidden_channel, out_channel, kernel_size=1, bias=True):
        super().__init__()
        self.hidden_channel = hidden_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias
        self.conv = nn.Conv2d(in_channels=self.hidden_channel,
                              out_channels=self.out_channel * 3,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)
        self.default_c_list = [self.out_channel] * 3

    def forward(self, input_tensor, cur_state, c_list=None):
        ''' 
        cc_i + cc_f + cc_g <--> input_tensor + cur_state
        cc_f <--> c_cur <--> cc_i <--> cc_g <--> c_next
        c_cur: long-term memory, last feature
        h_cur: short-term memory, g_buffer
        input_tensor: current feature(input)
        c_next: current feature(output)
        '''
        # log.debug(dict_to_string(input_tensor))
        combined = torch.cat(input_tensor, dim=1)  # concatenate along channel axis
        combined_conv = self.conv(combined)
        if c_list is None:
            c_list = self.default_c_list
        cc_i, cc_f, cc_g = torch.split(combined_conv, c_list, dim=1) # type: ignore 
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        # o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * cur_state + i * g
        # h_next = o * torch.tanh(c_next)

        return c_next
