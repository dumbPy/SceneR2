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

