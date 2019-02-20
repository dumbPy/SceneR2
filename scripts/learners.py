if __name__=="__main__":
    from core import *
else:
    from .core import *
class ModelLearner(nn.Module):
    """ModelLearner class takes model (initialized), loss function(not initializzed) and learning rate.
    Given each sample (x,y), it trains on it
    call epochEnded at the end of each epoch,
            passing parentLearnerClassObject that has trainLoader as its attribute

    Supports only Cross Entropy Loss and MSE Loss for now.
    Make sure the labels are not onehot encoded and you input 'classes' argument correctly.

    """
    def __init__(self,model, loss_fn:"don't Init", lr, optim:"dont' Init", modelName:str, Train=True, is_multi=True, classes=3,is_depth=False,*args, **kwargs):
        super().__init__()
        self.loss=loss_fn().to(device)
        self.lr=lr
        self.model=model.to(device)
        self.optim=optim(self.model.parameters(), self.lr)
        self.modelName=modelName
        self.is_depth=is_depth
        self.args,self.kwargs=args,kwargs
        self.train_epoch_loss=[]     #Add loss here for each batch and reset at end of epoch
        self.test_epoch_loss=[]      #same as above for test
        self.num_samples_seen=0
        self.Train=Train            #Training Mode Flag
        self.train_loss_list=[]     #to be updated at the end of  each epoch
        self.test_loss_list=[]
        self.is_multi=is_multi
        self.classes=classes
        self.train_confusion_matrix_list=[]
        self.valid_confusion_matrix_list=[]
        self.loss_name=self.loss.__class__.__name__
        if is_multi: 
            self.train_confusion_matrix=ConfusionMeter(classes)
            self.valid_confusion_matrix=ConfusionMeter(classes)
        if isinstance(self.loss, nn.MSELoss): 
            self.hot=oneHot(classes) #one hot encoder class to be used before feeding y to loss
        self.to(device)
        
        
        
    def forward(self, x, y):
        y_pred = self.model(x)
        if isinstance(self.loss, nn.CrossEntropyLoss): #Handeling specific requirements of CE Loss
            y=y.view(self.parentLearner.trainLoader.batch_size)
            y=y.long()
        if self.is_depth: #reshaping the y_pred and y before passing into loss if depth channel is present
            b,d=x.shape[0],x.shape[1]
            y_pred=y_pred.view(b*d, *y_pred.shape[2:])
            y=y.view(b*d, *y.shape[2:])
        if isinstance(self.loss, nn.MSELoss): y=self.hot(y)
        loss = self.loss(y_pred, y)
        if self.Train==True:
            self.num_samples_seen= self.num_samples_seen + x.shape[0]
            self.train_epoch_loss.append(loss.item())
            if len(y_pred.shape)>2: #Squish the batch, depth into batch
                y_pred=y_pred.view(np.asarray(y_pred.shape[:-1]).prod(), y_pred.shape[-1])
                y=y.view(np.asarray(y.shape[:-1]).prod(), y.shape[-1])
            self.train_confusion_matrix.add(y_pred.data, y.data)
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
        else: #Test Loop
            if len(y_pred.shape)>2: #Squish the batch, depth into batch
                y_pred=y_pred.view(np.asarray(y_pred.shape[:-1]).prod(), y_pred.shape[-1])
                y=y.view(np.asarray(y.shape[:-1]).prod(), y.shape[-1])
            self.valid_confusion_matrix.add(y_pred.data, y.data)
            self.test_epoch_loss.append(loss.item())


    def setTest(self):   self.Train=False; self.model.eval()
    def setTrain(self):  self.Train=True;  self.model.train()
    def save(self, model_name=None):
        if model_name is None: model_name=self.modelName
        self.model.save_state_dict(f"saved_models/{model_name}_lr{self.lr}/\
    loss_{self.loss_name}_epoch_{len(self.train_loss_list)}.pt")
    
        #setParent will give the ModelLearner access the higherlevel class attribures like trainLoader's length
    #and batch size, currentEpoch, etc
    def setParent(self, parentLearner): self.parentLearner=parentLearner
    def trainEpochEnded(self): 
        self.train_loss_list.append(np.asarray(self.train_epoch_loss).mean())
        self.train_epoch_loss=[]  #reset total_average_loss at the end of each epoch
        self.train_confusion_matrix_list.append(self.train_confusion_matrix.value().copy())
        self.train_confusion_matrix.reset()
        try: 
            epochs=self.parentLearner.epochsDone
            printEvery=self.parentLearner.printEvery
            if epochs%printEvery==0:    
                print(f"lr: {self.lr}      trainLoss: {self.train_loss_list[-1]}")
        except: print(f"lr: {self.lr}      trainLoss: {self.train_loss_list[-1]}")
    def testEpochEnded(self):
        self.test_loss_list.append(np.asarray(self.test_epoch_loss).mean())
        self.test_epoch_loss=[]
        self.valid_confusion_matrix_list.append(self.valid_confusion_matrix.value().copy())
        self.valid_confusion_matrix.reset() #reset confusion matrix for next epoch after appending it's values to the list above
        try: 
            epochs=self.parentLearner.epochsDone
            printEvery=self.parentLearner.printEvery
            if epochs%printEvery==0:
                print(f"lr: {self.lr}      {self.loss_name}validationLoss: {self.test_loss_list[-1]}")
        except: print(f"validationLoss: {self.loss_name}{self.test_loss_list[-1]}")
    @property
    def avg_loss(self): return 0 if len(self.train_epoch_loss)<1 else np.asarray(self.train_epoch_loss).mean()

class ParallelLearner(nn.Module):
    """ParallelLearner takes list of ModelLearners to be trained parallel on the same data samples
    from the passed pytorch dataLoader object. epochs are the number of epochs to be trained for
    """
    def __init__(self, listOfLearners, epochs=None, trainLoaderGetter=None, trainLoader=None, printEvery=math.inf, validLoader=None, validLoaderGetter=None, *args, **kwargs):
        super().__init__()
        self.learners=listOfLearners
        self.trainLoader=trainLoader
        self.trainLoaderGetter=trainLoaderGetter
        self.epochs=epochs
        self.args,self.kwargs = args,kwargs
        self.validLoader=validLoader           #trainLoader for test set
        self.validLoaderGetter=validLoaderGetter
        self.epochsDone=0  #epoch counter
        self.printEvery=printEvery #print every n epochs
        try: [learner.setParent(self) for learner in self.learners] #set self as parent of all ModelLearners
        except: print("Couldn't set ParallelLearner as parent of ModelLearners!!! Make sure you wrap models in ModelLearner instances")
        if not trainLoader is None: self.trainLoaderGetter=lambda: [self.trainLoader]
        if not validLoader is None: self.validLoaderGetter=lambda: [self.validLoader]
    
    def train(self, epochs):
        self.epochs=epochs
        startTime=time.time()
        for t in tqdm_notebook(range(self.epochs)):
            [learner.setTrain() for learner in self.learners] #set all ModelLearners to Train Mode
            for self.num_trainLoader, trainLoader in enumerate(self.trainLoaderGetter()):
                bar=tqdm(trainLoader, leave=False)
                for idx, (x,y) in enumerate(bar):
#                     x = x.view(self.trainLoader.batch_size,28*28).to(device)
#                     y = y.view(self.trainLoader.batch_size, 1).float().to(device)
                    x = x.float().to(device)
                    y = y.float().to(device)
                    [learner(x,y) for learner in self.learners]
                    bar.set_description(f"Avg Loss: {self.learners[0].avg_loss}") #set the progress bar's description to average loss
            self.epochsDone+=1
            if self.epochsDone%self.printEvery==0:
                print()
                print("*"*50)
                print(f"Epoch: {t}   Time Elapsed: {time.time()-startTime}")
            [learner.trainEpochEnded() for learner in self.learners]
            [learner.setTest() for learner in self.learners] #Set all ModelLearners to Test Model
            for self.num_validLoader, validLoader in enumerate(self.validLoaderGetter()):
                bar = tqdm(validLoader, leave=False)
                for idx, (x,y) in enumerate(bar):
                    x = x.float().to(device)
                    y = y.float().to(device)
                    [learner(x,y) for learner in self.learners]
                    bar.set_description(f"Avg Loss: {self.learners[0].avg_loss}") #set the progress bar's description to average loss
            [learner.testEpochEnded() for learner in self.learners]
        #Pass self to all learners defined above so they can use self.trainLoader to calculate it's total_loss before resetting epoch_loss
    
    
    def plotLoss_old(self, title, listOfLabelsForTrain, listOfLabelsForTest=None, xlabel="Epochs", ylabel="Loss", save=False):
        """Parameters:
        listOfLabelsForTrain: Labels for the train epoch loss for each ModelLearner
        listOfLabelsForTest : Labels for the test epoch loss for each ModelLearner, \
                              to be provided if validLoader was used to calculate loss on validation dataset.
        """
        assert len(listOfLabelsForTrain)==len(self.learners), "Provide Description for all Learners to Plot"
        import matplotlib.pyplot as plt
        import os
        # plt.switch_backend('agg') #should be uncommented when running unintractively on Cray Supercomputer 
        x=range(1,self.epochsDone+1)
        for i,learner in enumerate(self.learners):
            plt.plot(x, learner.train_loss_list, label=listOfLabelsForTrain[i])
        if (not (listOfLabelsForTest is None)) and (not (self.validLoader is None)):
            assert len(listOfLabelsForTest)==len(self.learners), \
                        "length of ListOfLabelsForTest is not same as num of learners"
            for i,learner in enumerate(self.learners):
                plt.plot(x, learner.test_loss_list, label=listOfLabelsForTest[i])
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.show(block=False)
        if save:
            os.makedirs("plots", exist_ok=True)
            plt.savefig(os.path.join("plots", title+".png"))
    
    def plotLoss(self, title="Loss Plot", listOfLabelsForTrain=[], listOfLabelsForTest=[], xlabel="Epochs", ylabel="Loss", save=False, subplots=True, num_cols=2, **kwargs):
        """Parameters:
        listOfLabelsForTrain: Labels for the train epoch loss for each ModelLearner
        listOfLabelsForTest : Labels for the test epoch loss for each ModelLearner, \
                              to be provided if validLoader was used to calculate loss on validation dataset.
        """
        if subplots==False: 
            return self.plotLoss_old(title, listOfLabelsForTrain, 
            listOfLabelsForTest, xlabel=xlabel, ylabel=ylabel, save=save)
        if len(self.learners)==1: 
            f,axes=plt.subplots(nrows=1, ncols=1, **kwargs)
            axes=np.asarray([axes])
        else:
            f,axes=plt.subplots(nrows=math.ceil(len(self.learners)/num_cols), ncols=num_cols, **kwargs)
        if len(axes.flat)>len(self.learners): f.delaxes(axes.flat[-1])
        x=range(1,self.epochsDone+1)
        for i,learner in enumerate(self.learners):
            axes.flat[i].plot(x, learner.train_loss_list, label= listOfLabelsForTrain[i] if i<len(listOfLabelsForTrain) else "Train Loss")
            axes.flat[i].plot(x, learner.test_loss_list, label=listOfLabelsForTest[i] if i<len(listOfLabelsForTest) else "Test Loss")
            axes.flat[i].set_title(learner.loss_name)
            axes.flat[i].set_xlabel(xlabel)
            axes.flat[i].set_ylabel(ylabel)
            axes.flat[i].legend()
        if not title is None: f.suptitle(title)
        # plt.xlabel(xlabel)
        # plt.ylabel(ylabel)
        # plt.legend()
        plt.show(block=False)
        if save:
            os.makedirs("plots", exist_ok=True)
            plt.savefig(os.path.join("plots", title+".png"))