if not __name__=="__main__":
    from .core import *
else:
    from core import *

class C3D(nn.Module):
    def __init__(self, in_channels:int, out_channels:list,
         kernel_size, padding=None, activation=nn.ReLU,
         batchNorm=False):
        super().__init__()
        layers=[]
        if isinstance(out_channels, int): out_channels=[out_channels] #single layer
        for out in out_channels:
            layers.append(nn.Conv3d(in_channels, out, kernel_size, padding=padding))
            if batchNorm: layers.append(nn.BatchNorm3d(out))
            layers.append(activation()) #adding activation layer after each Conv3D layer
            layers.append(TimeDistributed(nn.MaxPool2d((2,2))))    
            in_channels=out                  #input channels of next layer
        self.layers=nn.Sequential(*layers)

    def forward(self, x):
        x=x.permute(0,2,1,3,4)
        x= self.layers(x)
        x=x.permute(0,2,1,3,4)
        return x


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
        if len(x.shape)>1: #TimeDistributed(LossFunction) will return singular value
            newShape=list(x.shape)
            x=x.view(*oldShape[0:2]+newShape[1:])
        return x


class ConvLSTMCell(nn.Module):
    """
    Generate a convolutional LSTM cell
    Copied from https://github.com/Atcold/pytorch-CortexNet/blob/master/model/ConvLSTMCell.py
    """

    def __init__(self, in_channels, hidden_channels, kernel_size=3, padding=1):
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
        in_gate = torch.sigmoid(in_gate)
        remember_gate = torch.sigmoid(remember_gate)
        out_gate = torch.sigmoid(out_gate)

        # apply tanh non linearity
        cell_gate = torch.tanh(cell_gate)

        # compute current cell and hidden state
        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)
        hidden = out_gate * torch.tanh(cell)

        return hidden, cell

class ConvLSTM2D(nn.Module):
    def __init__(self, in_channels, out_channels=None, hidden_channels=None, bidirectional=False, kernel_size=3, return_sequence=True, padding=1, output_callback=None):
        """ConvLSTM2D implementation with help from Keras
        see https://keras.io/layers/recurrent/#convlstm2d for output shape w.r.t return_sequence
        Also added BiDirectional that is not available in Keras.
        Arguments:
        ----------------
        hidden_channels: (Int)  (default=out_channels) channels to use in hidden state, if you want to use them different from output_channels
        bidirectional  : (Bool) Bidirectional ConvLSTM2D Returned channels would be 2 x out_channels
        return_sequence: (Bool) Return the sequence of hidden states at each time step.
        output_callback: callback function to be called after each timestep and pass output as argument to it.
                         Only passes output of forward ConvLSTM. For Bidir, return sequence of hidden states and apply output_conv externally
                         output_callback can be used to avoid returning sequence of all hidden states, and pass the output_state to passed
                         function at each time step. (Not used in this project. Extra feature for future)
        ------------
        Returns: 
                output:        4D Tensor (batch,         directions*out_channels, out_height, out_width)
                hidden_states: 5D Tensor (batch, time, directions*hidden_channel, out_height, out_width)     only if return_sequence=True
        example:
                out, hidden_states=ConvLSTM2D(in_channels, out_channels, hidden_channels,return_sequence=True)
                out               =ConvLSTM2D(in_channels, out_channels, hidden_channels,return_sequence=False)


        If hidden_channels is passes, the hidden states are of that channels,
        return_sequence=True: Hidden States is size (batch, time, hidden_channels*direction, height, width) are returned
        """
        super().__init__()
        assert out_channels or hidden_channels, "At Least pass out_channel for last layer or hiden_channel for hidden states. None passed"
        self.in_channels=in_channels
        self.out_channels=out_channels
        if hidden_channels is None: self.hidden_channels=out_channels
        else: self.hidden_channels=hidden_channels
        self.kernel_size=kernel_size
        self.return_seq=return_sequence
        self.bidir=bidirectional
        self.out_callback=output_callback
        self.cell=ConvLSTMCell(self.in_channels, self.hidden_channels, kernel_size=kernel_size, padding=padding)
        #hidden_channels to out_channels. Will be applied to all hidden states only of out_callback is passed, else to last hidden state only
        if out_channels: self.outputLayer=nn.Conv2d(self.hidden_channels, self.out_channels, kernel_size=3, padding=1)
        else: self.outputLayer = lambda x: x #return same if no out_channel is padded. User will have to add hidden-->out_channel layer externaly

    def forward(self, x): #x of shape (batch, time, in_channel, height, width)
        states=[]       #states
        priv_state=None #at the start of sequence, previous state is None
        for t in range(x.shape[1]):
            xt=x[:,t,:,:,:]   #shape=(batch, in_channel, height, width) as single frame t is selected
            priv_state=self.cell(xt, priv_state) #returned ht,ct are the previous state for the next time frame
            if self.return_seq: states.append(priv_state[0])         #add ht to states to be returned if return_sequence=True
            if self.out_callback: self.out_callback(self.outputLayer(priv_state[0])) #pass each time step output to callback
        #add time dimension to all states and concat them to get 1 tensor of shape (batch, time, hidden_channel, out_height, out_width)

        if self.bidir:
            reverse_states=[]       #states
            reverse_priv_state=None #at the start of sequence, previous state is None
            for t in range(x.shape[1]-1, -1, -1):  #reverse sequence
                xt=x[:,t,:,:,:]   
                reverse_priv_state=self.cell(xt, reverse_priv_state) 
                if self.return_seq: reverse_states.append(reverse_priv_state[0])
            
            final_output=self.outputLayer(torch.cat([priv_state[0], reverse_priv_state[0]], dim=1)) #shape=(batch,direction*out_channels, out_height, out_width)
            if self.return_seq:
                #add time dimention to all states and concat them at time dimention
                states=torch.cat([state.unsqueeze_(1) for state in states], dim=1)
                reverse_states=torch.cat([state.unsqueeze_(1) for state in reverse_states], dim=1)
                #now concat the bidirectional states at channel dimention 
                # to get output shape as (batch, time, direction*hidden_channels, out_height, out_width)
                return (final_output,torch.cat([states, reverse_states], dim=2))

            else: #if return_sequence=False
                return final_output
        #Else Unidir ConvLSTM2D
        final_output=self.outputLayer(priv_state[0])   #output withhidden_channels --> out_channels from last hidden state if unidir convLSTM
        if self.return_seq: 
            states=torch.cat([state.unsqueeze_(1) for state in states], dim=1) #unsqueeze and concat hidden states at time dimension
            return final_output, states
        else: return final_output  #return last hidden state's output if return_states=False


