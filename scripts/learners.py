from .core import *
class modelLearner(nn.Module):
    """modelLearner class takes model, loss function and learning rate.
    Given each sample (x,y), it trains on it
    call epochEnded at the end of each epoch,
            passing parentLearnerClassObject that has trainLoader as its attribute

    Supports only Cross Entropy Loss and MSE Loss for now.

    """
    def __init__(self,model, loss_fn, lr, optim, modelName, Train=True, is_multi=True, classes=3,*args, **kwargs):
        super().__init__()
        self.loss=loss_fn().to(device)
        self.lr=lr
        self.model=model.to(device)
        self.optim=optim(self.model.parameters(), self.lr)
        self.modelName=modelName
        self.args,self.kwargs=args,kwargs
        self.train_epoch_loss=0     #Add loss here for each batch and reset at end of epoch
        self.test_epoch_loss=0      #same as above for test
        self.num_samples_seen=0
        self.Train=Train            #Training Mode Flag
        self.train_loss_list=[]     #to be updated at the end of  each epoch
        self.test_loss_list=[]
        self.is_multi=is_multi
        if is_multi: self.confusion_matrix=tnt.meter.ConfusionMeter(classes)
        if isinstance(self.loss, nn.MSELoss): self.loss_name="MSE "
        else: self.loss_name="CE "
        self.to(device)
        
        
    def forward(self, x, y):
        y_pred = self.model(x)
        if self.Train==True:
            if isinstance(self.loss, nn.CrossEntropyLoss): #Handeling specific requirements of CE Loss
                y=y.view(self.parentLearner.trainLoader.batch_size)
                y=y.long()
            loss = self.loss(y_pred, y)
            self.num_samples_seen= self.num_samples_seen + x.shape[0]
            self.train_epoch_loss += loss.item()
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
        else: #Test Loop
            if isinstance(self.loss, nn.CrossEntropyLoss): #Handeling specific requirements of CE Loss
                y=y.view(self.parentLearner.validLoader.batch_size)
                y=y.long()
            loss = self.loss(y_pred, y)
            self.confusion_matrix.add(y_pred.data, y.data)
            self.test_epoch_loss+= loss.item()


    def setTest(self):   self.Train=False
    def setTrain(self):  self.Train=True
    def save(self): self.model.save_state_dict(f"saved_models/{self.modelName}_lr{self.lr}/\
    loss_{self.loss_name}_epoch_{len(self.train_loss_list)}.pt")
    
        #setParent will give the modelLearner access the higherlevel class attribures like trainLoader's length
    #and batch size, currentEpoch, etc
    def setParent(self, parentLearner): self.parentLearner=parentLearner
    def trainEpochEnded(self): 
        try:    self.train_loss_list.append(self.train_epoch_loss/
                                            (len(self.parentLearner.trainLoader)*self.parentLearner.num_trainLoader+1))
        except: self.train_loss_list.append(self.train_epoch_loss/len(self.parentLearner.trainLoader))

        self.train_epoch_loss=0  #reset total_average_loss at the end of each epoch
        try: 
            epochs=self.parentLearner.epochsDone
            printEvery=self.parentLearner.printEvery
            if epochs%printEvery==0:    
                print(f"lr: {self.lr}      trainLoss: {self.train_loss_list[-1]}")
        except: print(f"lr: {self.lr}      trainLoss: {self.train_loss_list[-1]}")
    def testEpochEnded(self):
        try:    self.test_loss_list.append(self.test_epoch_loss/
                                           (len(self.parentLearner.validLoader)*self.parentLearner.num_validLoader+1))
        except: self.test_loss_list.append(self.test_epoch_loss/len(self.parentLearner.validLoader))
        self.test_epoch_loss=0
        try: 
            epochs=self.parentLearner.epochsDone
            printEvery=self.parentLearner.printEvery
            if epochs%printEvery==0:
                print(f"lr: {self.lr}      {self.loss_name}testLoss: {self.test_loss_list[-1]}")
        except: print(f"testLoss: {self.loss_name}{self.test_loss_list[-1]}")


class ParallelLearner(nn.Module):
    """ParallelLearner takes list of modelLearners to be trained parallel on the same data samples
    from the passed pytorch dataLoader object. epochs are the number of epochs to be trained for
    """
    def __init__(self, listOfLearners, epochs, trainLoaderGetter=None, trainLoader=None, printEvery=10, validLoader=None, validLoaderGetter=None, *args, **kwargs):
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
        try: [learner.setParent(self) for learner in self.learners] #set self as parent of all modelLearners
        except: print("Couldn't set ParallelLearner as parent of modelLearners!!!")
    
    
    def train(self):
        startTime=time.time()
        for t in range(self.epochs):
            [learner.setTrain() for learner in self.learners] #set all modelLearners to Train Mode
            for self.num_trainLoader, self.trainLoader in enumerate(self.trainLoaderGetter()):
                for idx, (x,y) in enumerate(self.trainLoader):
#                     x = x.view(self.trainLoader.batch_size,28*28).to(device)
#                     y = y.view(self.trainLoader.batch_size, 1).float().to(device)
                    x = x.float().to(device)
                    y = y.float().to(device)
                    [learner(x,y) for learner in self.learners]
            self.epochsDone+=1
            if self.epochsDone%self.printEvery==0:
                print()
                print("*"*50)
                print(f"Epoch: {t}   Time Elapsed: {time.time()-startTime}")
            [learner.trainEpochEnded() for learner in self.learners]
            if (not self.validLoaderGetter is None): #This part runs only when validLoaderGetter is provided
                [learner.setTest() for learner in self.learners] #Set all modelLearners to Test Model
                for self.num_validLoader, self.validLoader in enumerate(self.validLoaderGetter()):
                    for idx, (x,y) in enumerate(self.validLoader):
                        x = x.float().to(device)
                        y = y.float().to(device)
                        [learner(x,y) for learner in self.learners]
                [learner.testEpochEnded() for learner in self.learners]
        #Pass self to all learners defined above so they can use self.trainLoader to calculate it's total_loss before resetting epoch_loss
    
    
    def plotLoss(self, title, listOfLabelsForTrain, listOfLabelsForTest=None, xlabel="Epochs", ylabel="Loss", save=False):
        """Parameters:
        listOfLabelsForTrain: Labels for the train epoch loss for each modelLearner
        listOfLabelsForTest : Labels for the test epoch loss for each modelLearner, \
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
        os.makedirs("plots", exist_ok=True)
        plt.savefig(os.path.join("plots", title+".png"))