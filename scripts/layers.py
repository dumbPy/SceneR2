from .core import *

class C3D(nn.Module):
    def __init__(self, in_channels:int, out_channels:Union[list, int], kernel_size:Union[tuple,int], padding='same', activation=nn.ReLU):
        super().__init__()
        layers=[]
        if padding=='same': padding=kernel_size-1 #assuming dilation=1
        if isinstance(out_channels, int): out_channels=[out_channels] #single layer
        for out in out_channels:
            layers.append(nn.Conv3d(in_channels, out, kernel_size, padding=padding))
            layers.append(activation()) #adding activation layer after each Conv3D layer
            in_channels=out                  #input channels of next layer
        self.layers=nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class TimeDistributed(nn.Module):
    """Mimics the function of TimeDistributed wrapper as in Keras"""
    def __init__(self, module):
        super().__init__()
        self.module=module

    def forward(self, x): #expecting x to be of shape (batch_size, timesteps, ...)
        batch_size=x.shape[0]
        timesteps=x.shape[1]
        oldShape=list(x.shape.cpu().numpy())
        newShape=[batch_size*timesteps]+list(x.shape.cpu().numpy())[2:]
        x=x.view(*newShape)
        x=self.module(x)
        x=x.view(*oldShape)
        return x


class ConvLSTMCell(nn.Module):
    """
    Generate a convolutional LSTM cell
    Copied from https://github.com/Atcold/pytorch-CortexNet/blob/master/model/ConvLSTMCell.py
    """

    def __init__(self, input_size, hidden_size, kernel_size=3, padding=2):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.Gates = nn.Conv2d(input_size + hidden_size, 4 * hidden_size, kernel_size, padding=padding)

    def forward(self, input_, prev_state=None):

        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            prev_state = (
                Variable(torch.zeros(state_size)),
                Variable(torch.zeros(state_size))
            )

        prev_hidden, prev_cell = prev_state

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat((input_, prev_hidden), 1)
        gates = self.Gates(stacked_inputs)

        # chunk across channel dimension
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        # apply sigmoid non linearity
        in_gate = F.sigmoid(in_gate)
        remember_gate = F.sigmoid(remember_gate)
        out_gate = F.sigmoid(out_gate)

        # apply tanh non linearity
        cell_gate = F.tanh(cell_gate)

        # compute current cell and hidden state
        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)
        hidden = out_gate * F.tanh(cell)

        return hidden, cell