from .core import *
from .layers import *

class C3D_resnet_ConvLSTM2D(nn.Module):
    def __init__(self, n_classes, input_channels=3, adaptivePoolSize=None, input_shape=(224,224), dropout_p=0.5):
        super().__init__()
        if adaptivePoolSize: self.adaptivePool,self.adaptivePoolSize = True, adaptivePoolSize
        if isinstance(input_shape, int): self.input_shape=(input_shape,input_shape)
        else: self.input_shape=input_shape
        self.n_classes=n_classes
        try: #if fastai is installed, use fastai's resnext50. else, use torchvision's resnet50
            import fastai.model
            resnet=fastai.model.resnext50(pre=True)
        except: resnet=torchvision.models.resnet50(pretrained=True)

        res_layers=resnet.children()[:8] #Will gave output 2048x7x7 for an input of 3x224x224
        res_layers+=nn.Conv2d(2048, 256, kernel_size=1) #1x1 conv to shrink channels to 256
        resnet=nn.Sequential(*res_layers)
        
        self.resnet=TimeDistributed(resnet)
        self.c3d=C3D(in_channels=input_channels, out_channels=[32,32,32,64,128], kernel_size=3)
        self.convLSTM2D_bidir=ConvLSTM2D(in_channels=256+128, out_channels=128, hidden_channels=256)
        self.convLSTM2D_final=ConvLSTM2D(in_channels=128+128, out_channels=64, hidden_channels=256)
        if self.adaptivePool: self.fc=nn.Linear(64, self.n_classes)
        else: 
            
            self.fc=nn.Sequential([nn.Linear(64*input_shape[0]*input_shape[1]/(32*32), 512),
                                    nn.Dropout2d(dropout_p),
                                    nn.Linear(512, 256),
                                    nn.Dropout2d(dropout_p),
                                    nn.Linear(256,self.n_classes)])


    def forward(self, x): #x of shape (batch_size,time, in_channels=3, height, width)
        #concat at 2nd dimention i.e., channels. c3d and resnet outputs
        #  are expected to have same dimentions except for output channels. resnet will have 256, c3d will have 128
        c3d_resnet_out=torch.cat((self.resnet(x), self.c3d(x)), dim=2) #concat at channel dimension
        bidir_out=self.convLSTM2D_bidir(c3d_resnet_out)
        output   =self.convLSTM2D_final(bidir_out)
        output   =nn.Dropout3d(output)
        if self.adaptivePool: output=nn.AdaptiveAvgPool3d((64,1,1))(output) #keep time_frames same, pool accross height and width
        output=torch.flatten(output, start_dim=2)  #flatten (channel,height,width), keep batch and time dimension
        return self.fc(output)
        

        #If adaptive_pool is not used, the input_size of images will be fized. For 224*224, 7x7 is the size of output
        #when flattened, it becomes num_channels*49=64*49=3136
        


