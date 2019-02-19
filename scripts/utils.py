if  __name__=="__main__":
    from core import *
else:
    from .core import *

try: from fastai.dataset import BaseDataset
except: from torch.utils.data import Dataset as BaseDataset 



class oneVidNumpyDataset(BaseDataset):
    def __init__(self, fileName:str,resolution:tuple=(224,224), skipStart=125, skipEnd=125, get_label=None)->BaseDataset:
        """Dataset for extracting frames from passed .npy file and corrosponding labels from get_label
        """
        self.fileName=fileName
        self.get_label=get_label
        self.skipStart=skipStart
        self.skipEnd=skipEnd
        self.res=resolution
    
    def __getitem__(self,idx:int)->tuple:
        #skipping the first and last 125 frames.
        idx+=self.skipStart
        x=np.load(self.fileName)[idx]
        x=np.squeeze(x)     #extra 1 dimention removed if any
        if len(x.shape)==2 : x=np.dstack((x,x,x)) #id image is 1 channel, make it 3 channel
        x=np.rollaxis(x, 2)   #Make the images channel first
        x=x[:,:224,:224].astype("float32")
        return (x, self.get_label(self.fileName, idx))
    
    def get_y(self, idx):
        return self.get_label(self.fileName, idx+self.skipStart)
    
    def __len__(self):
        return len(np.load(self.fileName))-self.skipStart-self.skipEnd


class ConcatDatasetWithWeights(torch.utils.data.ConcatDataset):
    def __init__(self, fileNames, get_label):
        self.fileNames=fileNames
        self.get_label=get_label
        datasets=[oneVidNumpyDataset(fileName, get_label=get_label) for fileName in fileNames]
        super().__init__(datasets)
    
    @staticmethod
    def fixLabelLength(labels, dataset):
        """If labels of some trailing frames are missing, broadcast last label to all trailing frames
        If labels are more than frames, remove the extra labels from end"""
        labels=list(labels)
        while len(labels)<len(dataset):
            labels+=labels[-1]
        while len(labels)>len(dataset):
            labels.pop() #remove last extra element
        return labels
    
    def get_y(self, idx):
        import bisect
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx].get_y(sample_idx)
    
    def getWeights(self):
        try: _=self.weights
        except:
            allLabels=[self.get_y(i) for i in range(self.__len__())]
            classes=set(allLabels)                     #will return all classes
            counts = [allLabels.count(someClass) for someClass in classes]
            self.weights=[1/counts[label] for label in allLabels] #convert labels to weights as weight=1/count[label]
        return self.weights 

class dumbWeightedRandomSampler(torch.utils.data.Sampler):
    """weightedRandomSampler like torch.utils.data.weightedRandomSampler but with numpy insices rather than torch"""
    def __init__(self, weights, num_samples, replacement=True):
        from torch._six import int_classes as _int_classes
        if not isinstance(num_samples, _int_classes) or isinstance(num_samples, bool) or \
                num_samples <= 0:
            raise ValueError("num_samples should be a positive integeral "
                             "value, but got num_samples={}".format(num_samples))
        if not isinstance(replacement, bool):
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(replacement))
        self.weights = weights
        self.num_samples = num_samples
        self.replacement = replacement        
        
    def __iter__(self):
        import random
        return iter(random.choices(list(range(len(self.weights))), weights=self.weights, k=self.num_samples))

    def __len__(self):
        return self.num_samples


def vid_from_csv(filename, vid_folder=None):
    if vid_folder is None: 
        assert(os.path.exists(globalVariables.path_to_vids)),\
                "please update globalVariables.path_to_vids or pass vid_folder as argument. default path does not exists"
        vid_folder=globalVariables.path_to_100_vids
    vid_files=[vid_folder+filename for filename in os.listdir(vid_folder()) if filename.split(".")[-1] == 'avi']
    assert(len(vid_files)>0), "No Video Files Found in "+vid_folder()+" \
        make sure the location is mounted"
    for vidFile in vid_files:
        if SingleCSV.get_file_id(filename) in  vidFile: return vidFile
    raise NameError(SingleCSV.get_file_id(filename)+" not found in "+vid_folder())