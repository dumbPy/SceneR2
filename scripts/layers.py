from .core import *

class C3D(nn.Module):
    def __init__(self, in_channels:int, out_channels:list,
         kernel_size, padding='same', activation=nn.ReLU,
         batchNorm=False):
        super().__init__()
        layers=[]
        if padding=='same': padding=kernel_size-1 #assuming dilation=1
        if isinstance(out_channels, int): out_channels=[out_channels] #single layer
        for out in out_channels:
            layers.append(nn.Conv3d(in_channels, out, kernel_size, padding=padding))
            if batchNorm: layers.append(nn.BatchNorm3d(out))
            layers.append(activation()) #adding activation layer after each Conv3D layer
            layers.append(nn.MaxPool3d((1,2,2)))
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
        oldShape=list(x.shape)
        newShape=[batch_size*timesteps]+list(x.shape)[2:]
        x=x.view(*newShape)
        x=self.module(x)
        x=x.view(*oldShape)
        return x


class ConvLSTMCell(nn.Module):
    """
    Generate a convolutional LSTM cell
    Copied from https://github.com/Atcold/pytorch-CortexNet/blob/master/model/ConvLSTMCell.py
    """

    def __init__(self, in_channels, hidden_channels, kernel_size=3, padding=2):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.Gates = nn.Conv2d(in_channels + hidden_channels, 4 * hidden_channels, kernel_size, padding=padding)

    def forward(self, input_, prev_state=None):#Input shape (batch, channel, height, width)

        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_channels] + list(spatial_size)
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

class ConvLSTM2D(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=None, bidirectional=False, kernel_size=3, return_sequence=True, padding=2):
        """ConvLSTM2D implementation with help from Keras
        see https://keras.io/layers/recurrent/#convlstm2d for output shape w.r.t return_sequence
        Also added BiDirectional that is not available in Keras.
        ------------
        Returns: 5D Tensor (batch time, directions*out_channels, out_height, out_width)     if return_sequence=True
                 4D Tensor (batch,      directions*out_channels, out_height, out_width)     if return_sequence=False
        
        """
        super().__init__()
        self.in_channels=in_channels
        self.out_channels=out_channels
        if hidden_channels is None: self.hidden_channels=in_channels
        else: self.hidden_channels=hidden_channels
        self.kernel_size=kernel_size
        self.return_seq=return_sequence
        self.bidir=bidirectional
        self.cell=ConvLSTMCell(self.in_channels, self.hidden_channels, kernel_size=kernel_size, padding=padding)

    def forward(self, x): #x of shape (batch, time, in_channel, height, width)
        states=[]       #states
        priv_state=None #at the start of sequence, previous state is None
        for t in range(x.shape[1]):
            xt=x[:,t,:,:,:]   #shape=(batch, in_channel, height, width) as single frame t is selected
            priv_state=self.cell(xt, priv_state) #returned ht,ct are the previous state for the next time frame
            if self.return_seq: states.append(priv_state[0])         #add ht to states to be returned if return_sequence=True
        
        if self.bidir:
            reverse_states=[]       #states
            reverse_priv_state=None #at the start of sequence, previous state is None
            for t in range(x.shape[1]-1, 0, -1):  #reverse sequence
                xt=x[:,t,:,:,:]   
                reverse_priv_state=self.cell(xt, reverse_priv_state) 
                if self.return_seq: reverse_states.append(reverse_priv_state[0])
            
            if self.return_seq:
                #add time dimention to all states and concat them at time dimention
                states=        torch.cat([state.unsqueeze_(1) for state in         states], dim=1) 
                reverse_states=torch.cat([state.unsqueeze_(1) for state in reverse_states], dim=1)
                #now concat the bidirectional states at channel dimention 
                # to get output shape as (batch, time, direction*out_channels, out_height, out_width)
                return torch.cat([states, reverse_states], dim=2)

            else: #if return_sequence=False
                return torch.cat([priv_state[0], reverse_priv_state[0]], dim=1) #shape=(batch,direction*out_channels, out_height, out_width)
        
        if self.return_seq: return states
        else: return priv_state[0]  #return last hidden state if return_states=False


