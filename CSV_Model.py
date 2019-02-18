
# coding: utf-8

# In[1]:




# In[2]:


from scripts.core import *


# In[3]:


from scripts.dataset import CSVData, MovingObjectData, SingleCSV
from scripts.loss import weightedMSE
import pickle
from scripts.learners import ModelLearner, ParallelLearner

# In[45]:


# data=MovingObjectData.fromCSVFolder("/home/sufiyan/data/Daimler/100_vids/csv/")
# with open("/home/sufiyan/data/Daimler/100_vids/pickled_160_csv_14Feb", 'wb') as f:
#     pickle.dump(data, f)


# In[72]:


with open("/home/sufiyan/data/Daimler/100_vids/pickled_160_csv_14Feb", 'rb') as f:
    data=pickle.load(f)


# In[5]:


path="/home/sufiyan/Common_data/mtp2/dataset/NEW/100_vids/"


# In[73]:


# In[74]:


# len([1 for x,y in data if y==2])


# In[50]:


SingleCSV.fromCSV("/home/sufiyan/data/Daimler/100_vids/csv/20170305_012110_S235_500K_V173015_LW_DML.csv", supressPostABA=False).plot()


# Seperating train and test files.

# In[51]:


np.random.seed(5) #seting seed for always selecting same train and test set
totalFiles=len(data)
testFiles=np.random.choice(totalFiles, int(0.2*totalFiles)) #20% files as test files
trainSampler=torch.utils.data.RandomSampler([i for i in range(totalFiles) if not i in testFiles])
testSampler =torch.utils.data.RandomSampler([i for i in range(totalFiles) if i in testFiles])


# In[52]:


trainLoader=torch.utils.data.DataLoader(data, batch_size=1, sampler=trainSampler)
testLoader=torch.utils.data.DataLoader(data, batch_size=1, sampler=testSampler)

print(len([1 for x,y in testLoader if y==1]))

# In[53]:


class CSVmodel(nn.Module):
    def __init__(self,categories=3):
        super().__init__()
#         self.l0=nn.Linear(in_features=4, out_features=4)
        self.l1=nn.LSTM(input_size=4,hidden_size=10, dropout=0.1, num_layers=1)
        self.l2=nn.Linear(in_features=10,out_features=categories)
        self.softmax=nn.Softmax()
    def forward(self,x):
#         print('X1 Shape: ', x.shape)
#         x=self.l0(x)
        x,_=self.l1(x)
        x=x[:,-1,:]
#         print('X2 Shape: ', x.shape)
        x= self.l2(x)
#         print('X3 Shape: ', x.shape)
        return x


# In[54]:





# In[55]:


learner=ParallelLearner(
    [ModelLearner(CSVmodel(), lr=0.005, loss_fn=partial(weightedMSE, [0.1,5,0.1]), optim=torch.optim.SGD, modelName='movingObjectDataModel' )]
    , epochs=10, trainLoader=trainLoader, printEvery=100, validLoader=testLoader)


# In[56]:


learner.train(10)


# In[57]:


learner.plotLoss("Loss", ["trainLoss"], ["testLoss"])


# In[58]:


modLearner=learner.learners[0]
modLearner.setTest()


# In[68]:


# modLearner.num_samples_seen


# In[67]:


print(len([1 for x,y in data if y==0]))
print(len([1 for x,y in data if y==1]))
print(len([1 for x,y in data if y==2]))


# In[75]:


print("Training Confusion Matrix", modLearner.train_confusion_matrix_list[0])


# In[76]:


print("Validation Confusion Matrix: ", modLearner.valid_confusion_matrix_list[0])


# In[77]:


# for x,y in testLoader:
#     x=x.float().to(device)
#     x=modLearner.model(x).detach().cpu().numpy()
#     y=y.cpu().numpy()
#     print(np.argmax(x), y)


# Fix tqdm for validation and put the model on cpu

# In[27]:


# len([1 for x,y in trainLoader if y==0])


# # In[18]:


# len([y for x,y in trainLoader if y==0])


# In[19]:


# path="/home/sufiyan/Common_data/mtp2/dataset/NEW/100_vids/"
# print([filename for filename in os.listdir(path+"LEFT")])
# if  True in [self.file_id in filename for filename in os.listdir(path+"LEFT")]: print 0 #Left Class
# elif True in [self.file_id in filename for filename in  os.listdir(path+"RIGHT")]: print 1 #Right Class
# else: print 2 


# In[20]:

