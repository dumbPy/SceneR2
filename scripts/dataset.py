if not __name__=="__main__":
    from .core import *
else:
    from core import *

allCols=["ABA_typ_WorkFlowState OPC_typ_BrakeReq ABA_typ_ABAAudioWarn ABA_typ_SelObj BS_v_EgoFAxleLeft_kmh BS_v_EgoFAxleRight_kmh RDF_val_YawRate RDF_typ_ObjTypeOr RDF_dx_Or RDF_v_RelOr RDF_dy_Or RDF_typ_ObjTypeOs RDF_dx_Os RDF_v_RelOs RDF_dy_Os RDF_typ_ObjTypePed0 RDF_dx_Ped0 RDF_vx_RelPed0 RDF_dy_Ped0 "]
allCols=allCols[0].split()

class BaseObject:
    def __init__(self,cols, df, supressOutliers=True, supressPostABA=True, **kwargs):
        self.df=df.loc[1:,cols]
        self.kwargs=kwargs
        if 'name' in kwargs: self.name=kwargs['name']
        self.cols=cols
        self.full_df=df.loc[:,allCols]
        self.SupressOutliers(**self.kwargs)
        if supressPostABA: self.SupressPostABA()
        
    def SupressOutliers(self, outlier_limit=-50, **kwargs):
        # threshold the folating -200 or -250 values to 0 for better resolution in plotting
        # outlier floating value is different for pedestrian.
        # hence outlier limit of -50 does not work for pedestrian.
        # For pedestrians, when there is no detection, the value goes to around -12.5 
        # that needs to be corrected to 0.
        try: outlier_limit=self.kwargs["outlier_limit"]
        except:pass
        for col in self.cols:
            self.df[col]=self.df[col].apply(lambda x: 0 if x< outlier_limit else x)
        return self
        
    def getABAReactionIndex(self):
        index=self.full_df[self.full_df["ABA_typ_WorkFlowState"]>0]["ABA_typ_WorkFlowState"].index[1]
        return index

    def SupressPostABA(self, **kwargs):
        """Truncate df when ABA stops tracking relevant object post ABA"""
        #kwargs['edgePostABA'] is supplied by SingleCSV.__init__()
        self.df=self.df.iloc[:self.kwargs['edgePostABA'],:]
        return self
        
    def SupressCarryForward(self):
        """Supresses the values when 1st column "RDF_typ_ObjType**" is zero. 
        Then's when the ABA isn't detecting any aboect in it's class but the values as carried forward."""
        for i, row in self.df.iterrows():
            if row[self.df.columns[0]]==0:
                for c,col in enumerate(self.df.columns[1:]): self.df.iloc[i, c+1]=0
        return self
    
    @staticmethod
    def getEdges(column, threshold=0.5):
        """returns indices of edge where laplace gradient is greater than 1
            Can be used to detect sections of object tracking.
        """
        ar=np.asarray(column)
        return [i for i,v in enumerate(laplace(ar)>threshold) if v==True] 

    def getEdgePostABA(self, threshold=0.5):
        edges=self.getEdges(self.df.iloc[:,0], threshold=threshold)
        ABAReactionIndex=self.getABAReactionIndex()
        edges=[edge for edge in edges if edge>ABAReactionIndex]
        if len(edges)>0: return edges[0]
        else: return self.df.index[-1] #if not edge found, keep whole df


    def plot(self, ax=None, subplots=True, **kwargs):
        ax=self.df.plot(ax=ax,subplots=True, title=self.name, **kwargs)
        # ax.legend(bbox_to_anchor=(1.5, 1))
        # ax.set_title(self.name)
        return ax
        
class ABAReaction(BaseObject):
    def __init__(self, df, *args, **kwargs):
        self.cols=["ABA_typ_WorkFlowState", "OPC_typ_BrakeReq", "ABA_typ_ABAAudioWarn", "ABA_typ_SelObj"]
        #Not to supress ABA signals, but supress only CAN signals below
        try: kwargs["supressPostABA"]=False
        except: pass
        super().__init__(self.cols, df, *args, **kwargs, name='ABA_Reaction')    

class MovingObject(BaseObject):
    def __init__(self, df, *args, **kwargs):
        self.cols=["RDF_typ_ObjTypeOr", "RDF_dx_Or", "RDF_v_RelOr", "RDF_dy_Or"]
        super().__init__(self.cols, df, *args, **kwargs, name='MovingObj')

class StationaryObject(BaseObject):
    def __init__(self, df, *args, **kwargs):
        self.cols=["RDF_typ_ObjTypeOs", "RDF_dx_Os", "RDF_v_RelOs", "RDF_dy_Os"]
        super().__init__(self.cols, df, *args, **kwargs, name='StationaryObj')

class PedestrianObject(BaseObject):
    def __init__(self, df, *args, **kwargs):
        #anything beyond -5 is floating value for pedestrian (assumption based on plots)
        self.cols=["RDF_typ_ObjTypePed0", "RDF_dx_Ped0", "RDF_vx_RelPed0","RDF_dy_Ped0"]
        super().__init__(self.cols, df, outlier_limit=-5, *args, **kwargs, name='Pedestrain')

class SingleCSV(object):

    #All Objects that are trackable from Daimler CAN_data csv and are subclasses of BaseObject.
    allObjects=[ABAReaction, MovingObject, StationaryObject, PedestrianObject]
    
    def __init__(self, df:pd.DataFrame, filename, dataObjectsToUse:list=None, **kwargs):
        """
        df: Pandas dataFrame to be cleaned and parsed
        dataObjetsToUse: list of BaseObject subclasses to use to parse the dataFrame.
                         dafault [ABAReaction, MovingObject, StationaryObject, PedestrianObject]
        these dataObjectsToUse will be initialized (and internally cleaned) 
        and joined again to return with SingleCSV.df
        """
        
        self.kwargs=kwargs
        self.filename=filename
        if not dataObjectsToUse:
            dataObjectsToUse=[ABAReaction, MovingObject, StationaryObject, PedestrianObject]
        else: 
            for obj in dataObjectsToUse:
                if not obj in self.allObjects:
                    raise TypeError("dataObjects should be a list of BaseObject subclasses to use")
        #find relevant object to be used to get `edge after ABA reaction` to truncate df
        self.relevantObjectIndex=self.get_relevant_object(df)
        self.edgePostABA=self.allObjects[self.relevantObjectIndex](df, supressPostABA=False).getEdgePostABA()
        self.kwargs['edgePostABA']=self.edgePostABA
        self.allObjects=[obj(df, **self.kwargs) for obj in dataObjectsToUse]

    @property
    def df(self): #returns Dataframe
        if hasattr(self, "outData"): return self.outData
        maxIndex=np.min([cols.df.index[-1] for cols in self.allObjects])
        self.outData=self.allObjects[0].df.iloc[:maxIndex,:].copy() #To avoid nan values in supressed values
        for obj in self.allObjects[1:]: self.outData=self.outData.join(obj.df.iloc[:maxIndex,:])
        return self.outData
    @property
    def values(self): return self.df.values #numpy equivalent, returns numpy array of dataframe
    @property
    def data(self): return (self.values, self.label) #tuple of (X,y) for dataloader in numpy format
    @property
    def file_id(self): return "_".join(self.filename.split("/")[-1].split(".")[0].split("_")[:2]) #eg: 20170516_015909
    # @property
    # def label(self): #One Hot Encoded labels
    #     """
    #     Returns: [1,0,0] : Left
    #          [0,1,0] : Right
    #          [0,0,1] : Stop/Other
    #     """
    #     path="/home/sufiyan/Common_data/mtp2/dataset/NEW/100_vids/"
    #     if  True in [self.file_id in filename for filename in os.listdir(path+"LEFT")]: return np.asarray([1,0,0]) #Left Class
    #     elif True in [self.file_id in filename for filename in  os.listdir(path+"RIGHT")]: return np.asarray([0,1,0]) #Right Class
    #     else: return np.asarray([0,0,1]) #Other class

    
    @property
    def label(self): # Not hot encoded values. required this way for the learner classes in scripts.learners
        """
        Returns: 
             0 : Left
             1 : Right
             2 : Stop/Other
        """
        path="/home/sufiyan/Common_data/mtp2/dataset/NEW/100_vids/"
        if  True in [self.file_id in filename for filename in self.filesWithoutFlip(path+"LEFT")+self.filesWithFlip(path+"RIGHT")]: return 0 #Left Class
        elif True in [self.file_id in filename for filename in  self.filesWithoutFlip(path+"RIGHT")+self.filesWithFlip(path+"LEFT")]: return 1 #Right Class
        else: return 2 #Other class

    @staticmethod
    def filesWithoutFlip(path_to_folder):
        files=[filename for filename in os.listdir(path_to_folder) if filename.split('.')[0].split('_')[0]!='FLIP']
        return files
    
    @staticmethod
    def filesWithFlip(path_to_folder):
        files=[filename for filename in os.listdir(path_to_folder) if filename.split('.')[0].split('_')[0]=='FLIP']
        return files

    @classmethod
    def fromCSV(cls, filename, **kwargs):
        return cls(read_csv_auto(filename), filename=filename, **kwargs)
    
    relevantObjects={0:"Driving/Moving Object", 1:"Stationary Object", 2:"Pedestrian A", 3:"Pedestrian B"}

    @staticmethod
    def get_relevant_object(df):
        """return from {1:Moving Object, 2: Stationary Object, 3: PedestrianObject}
        returns index as per SingleCSV.allObjects above 
        rather than SingleCSV.relevantObjects dictionary defined just above this method"""
        ABA_ReactionIndex=df[df["ABA_typ_WorkFlowState"]>0]["ABA_typ_WorkFlowState"].index[1]
        relevantObjectIndex=df["ABA_typ_SelObj"][ABA_ReactionIndex]
        #Pedestrian A(2) and B(3) are same in tracking data hence,
        if relevantObjectIndex==3 : relevantObjectIndex=2
        return relevantObjectIndex+1 #as to skip ABAReaction Object in SingleCSV.allObjects
    
    @staticmethod
    def print_relevant_object(df):
        ABA_ReactionIndex=df[df["ABA_typ_WorkFlowState"]>0]["ABA_typ_WorkFlowState"].index[1]
        relevantObjectIndex=df["ABA_typ_SelObj"][ABA_ReactionIndex]
        print("Reason for Braking: ", SingleCSV.relevantObjects[relevantObjectIndex])

    def plot(self, **kwargs):
        self.print_relevant_object(self.df)
        for obj in self.allObjects: obj.plot(**kwargs)

    def show_as_image(self, ax=None):
        if ax==None: f,ax=plt.subplots(1,1)
        ax.imshow(self.values, extent=[0,10,0,10])
        plt.show()
        return ax

class CSVData(data.Dataset):
    def __init__(self, files_list, preload=True, **kwargs):
        self.files=files_list
        self.kwargs=kwargs
        super().__init__()
        self.preload=preload
        if preload: self.data = [SingleCSV.fromCSV(self.files[i], **self.kwargs).data for i in range(self.__len__())]

    def __len__(self): return len(self.files)
    
    def __getitem__(self, i):
        if self.preload: return self.data[i]
        else: return SingleCSV.fromCSV(self.files[i], **self.kwargs).data

    @classmethod
    def fromCSVFolder(cls, folder, **kwargs):
        files=[os.path.join(folder, file) for file in os.listdir(folder) if file.split(".")[-1]=='csv']
        return cls(files, **kwargs)


class MovingObjectData(CSVData):
    def __init__(self, files_list, **kwargs):
        kwargs["dataObjectsToUse"]=[MovingObject] #add dataObject to use rather than all objects
        super().__init__(files_list, **kwargs)
