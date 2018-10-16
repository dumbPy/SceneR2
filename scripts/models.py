# import sys, os
# file_dir = os.path.dirname(__file__)
# sys.path.append(file_dir)
from core import *
from layers import *

class C3D_resnet_ConvLSTM2D(nn.Module):
    def __init__(self, n_classes, input_channels=3, adaptivePoolSize=None, input_shape=(224,224), dropout_p=0.5):
        super().__init__()
        if adaptivePoolSize: self.adaptivePool,self.adaptivePoolSize = True, adaptivePoolSize
        else: self.adaptivePool=False
        if isinstance(input_shape, int): self.input_shape=(input_shape,input_shape)
        else: self.input_shape=input_shape
        self.n_classes=n_classes
        self.dropout_p=dropout_p
        try: #if fastai is installed, use fastai's resnext50. else, use torchvision's resnet50
            import fastai.model
            resnet=fastai.model.resnext50(pre=True)
        except: resnet=torchvision.models.resnet50(pretrained=True)

        res_layers=[layer for i,layer in enumerate(resnet.children()) if i<8] #Will gave output 2048x7x7 for an input of 3x224x224
        res_layers+=[nn.Conv2d(2048, 256, kernel_size=1)] #1x1 conv to shrink channels to 256
        resnet=nn.Sequential(*res_layers)
        
        self.resnet=TimeDistributed(resnet)
        self.c3d=C3D(in_channels=input_channels, out_channels=[32,32,32,64,128], kernel_size=3, padding=1)
        self.convLSTM2D_bidir=ConvLSTM2D(in_channels=int(256+128), hidden_channels=128, bidirectional=True)
        self.convLSTM2D_final=ConvLSTM2D(in_channels=int(128+128), hidden_channels=64)
        if self.adaptivePool: self.fc=nn.Linear(64, self.n_classes)
        else: 
            out_shape=np.asarray(input_shape)//2//2//2//2//2
            self.linear_input=int(64*np.multiply(*out_shape))
            self.fc=nn.Sequential(*[nn.Linear(self.linear_input, 512),
                                    nn.Dropout2d(dropout_p),
                                    nn.Linear(512, 256),
                                    nn.Dropout2d(dropout_p),
                                    nn.Linear(256,self.n_classes)])


    def forward(self, x): #x of shape (batch_size,time, in_channels=3, height, width)
        #concat at 2nd dimention i.e., channels. c3d and resnet outputs
        #  are expected to have same dimentions except for output channels. resnet will have 256, c3d will have 128
        out1, out2=self.resnet(x), self.c3d(x)
        #resnet output's height/width is 1 smaller than c3d for some input dimensions
        if list(out2.shape)[-1]+1==list(out1.shape)[-1]: out2=self.pad(out2)
        c3d_resnet_out=torch.cat((out1,out2), dim=2) #concat at channel dimension
        _, bidir_states   =self.convLSTM2D_bidir(c3d_resnet_out)
        _, final_states   =self.convLSTM2D_final(bidir_states)
        output   =nn.Dropout3d(self.dropout_p)(final_states)
        if self.adaptivePool: output=nn.AdaptiveAvgPool3d((64,1,1))(output) #keep time_frames same, pool accross height and width
        output=torch.flatten(output, start_dim=2)  #flatten (channel,height,width), keep batch and time dimension
        return self.fc(output)
        #If adaptive_pool is not used, the input_size of images will be fized. For 224*224, 7x7 is the size of output
        #when flattened, it becomes num_channels*49=64*49=3136
    @staticmethod
    def pad(tensor):
        """Pads the tensor with 1 zero pad on right and bottom dimension
        Used when resnet returns 1 dimension bigger than c3d in some size cases
        Not required for common sizes like (224,224) or (160,160)"""
        return TimeDistributed(nn.ZeroPad2d((0,1,0,1)))(tensor)


if __name__=="__main__":
    d=200
    model=C3D_resnet_ConvLSTM2D(3, 3, input_shape=(d,d), adaptivePoolSize=1).cuda()
    a=torch.randn((1,20,3,d,d)).cuda()
    y=model(a)
    print("!"*50)
    print(" "*10+"Test Passed")
    print(f"Output shape :{y.shape}")
    print("!"*50)