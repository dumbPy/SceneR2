from SceneR2.dataset import *
from SceneR2.models import *
import unittest



class TestModels(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.d = 224
        self.X = torch.randn((1,20,3,self.d,self.d), device=device)
        self.can  = torch.rand((1,20,4), device=self.device)
        
    
    def tearDown(self):
        del self.X
        del self.can

    def test_C3D_Resnet_ConvLSTM(self):
        d = self.d
        model=C3D_resnet_ConvLSTM2D(n_classes=3, input_shape=(d,d),
        adaptivePoolSize=1).to(self.device)
        y = model(self.X)
        self.assertTrue((np.asarray(y.shape)==np.asarray([1,20,3])).all())
    
    def test_ResLSTM(self):
        model = ResLSTM(3).to(self.device)
        y=model((self.X,self.can))
        self.assertTrue((np.asarray(y.shape)==np.asarray([1,1,3])).all())
    

if __name__=='__main__':

    unittest.main()