from .core import *

class weightedMSE(torch.nn.MSELoss):
    def __init__(self, weight, *args, **kwargs):
        self.weight=torch.from_numpy(np.asarray([weight])).float().to(device)

        super().__init__(*args, **kwargs)
    def forward(self, input, target):
        return (((input-target)**2)*self.weight.expand(target.shape)).mean()