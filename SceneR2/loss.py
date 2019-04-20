if  __name__=="__main__":
    from core import *
else:
    from .core import *
    
class weightedMSE(torch.nn.MSELoss):
    def __init__(self, weight, *args, **kwargs):
        self.weight=torch.from_numpy(np.asarray([weight])).float().to(device)

        super().__init__(*args, **kwargs)
    def forward(self, pred, target):
        return (((pred-target)**2)*self.weight.expand(target.shape)).mean()

class FocalMSE(torch.nn.MSELoss):
    def __init__(self,threshold=0.7, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.threshold=threshold

    def forward(self, pred, target):
        delta = pred-target
        return (((delta)**2)*(pred<self.threshold).float()).mean()

class FocalMultiClass(torch.nn.CrossEntropyLoss):
    """FocalMultiClass inspired by 'Focal Loss from Focal Loss for Dense Object Detection', Kaiming He, et.al; 
    except this one works for multiclass classification, not just binary
    """
    def __init__(self, weight=None, gamma=1, size_average=None, ignore_index=-100,
                 reduce=None, reduction='mean'):
        super().__init__(weight, size_average, ignore_index=ignore_index, 
                        reduce=reduce, reduction='none')
        self.gamma=gamma
        self.softmax = nn.Softmax(dim=-1)

    def forward(self,input, target):
        s=self.softmax(input)
        mult = 1- torch.tensor([s[i][target[i]] for i in range(s.shape[0])], 
                            device=device)
        ce = super().forward(input, target)
        return (mult**self.gamma*ce).mean()