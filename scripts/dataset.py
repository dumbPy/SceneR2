if  __name__=="__main__":
    from core import *
    from utils import *
else:
    from .core import *
    from .utils import *

allCols=["ABA_typ_WorkFlowState OPC_typ_BrakeReq ABA_typ_ABAAudioWarn ABA_typ_SelObj BS_v_EgoFAxleLeft_kmh BS_v_EgoFAxleRight_kmh RDF_val_YawRate RDF_typ_ObjTypeOr RDF_dx_Or RDF_v_RelOr RDF_dy_Or RDF_typ_ObjTypeOs RDF_dx_Os RDF_v_RelOs RDF_dy_Os RDF_typ_ObjTypePed0 RDF_dx_Ped0 RDF_vx_RelPed0 RDF_dy_Ped0 "]
allCols=allCols[0].split()

class BaseObject:
    def __init__(self,cols, df, supressOutliers=True, supressPostABA=True, supressCarryForward=False, **kwargs):
        self.df=df.loc[1:,cols]
        self.kwargs=kwargs
        if 'name' in kwargs: self.name=kwargs['name']
        self.cols=cols
        self.full_df=df.loc[:,allCols]
        self.SupressOutliers(**self.kwargs)
        if supressPostABA: self.SupressPostABA()
        if supressCarryForward: self.SupressCarryForward()
        
    def SupressOutliers(self, outlier_threshold=-50, **kwargs):
        # threshold the folating -200 or -250 values to 0 for better resolution in plotting
        # outlier floating value is different for pedestrian.
        # hence outlier limit of -50 does not work for pedestrian.
        # For pedestrians, when there is no detection, the value goes to around -12.5 
        # that needs to be corrected to 0, with threhold=-5.
        try: outlier_threshold=self.kwargs["outlier_threshold"]
        except:pass
        for col in self.cols:
            self.df[col]=self.df[col].apply(lambda x: 0 if x< outlier_threshold else x)
        return self
        
    def getABAReactionIndex(self):
        index=self.full_df[self.full_df["ABA_typ_WorkFlowState"]>0]["ABA_typ_WorkFlowState"].index[1]
        return index

    def SupressPostABA(self, **kwargs):
        """Truncate df when ABA stops tracking relevant object post ABA"""
        #kwargs['edgePostABA'] is supplied by SingleCSV.__init__()
        try: _=self.kwargs['edgePostABA']
        #would probably mess up as edgePostABA will be different in all subClasses.
        #here only so that I can use each subClass independently for debugging
        except: self.kwargs['edgePostABA']=self.getEdgePostABA()
        self.df=self.df.iloc[:self.kwargs['edgePostABA'],:]
        return self
        
    def SupressCarryForward(self):
        """Supresses the values when 1st column "RDF_typ_ObjType**" is zero. 
        Then's when the ABA isn't detecting any abject in it's class but the values as carried forward."""
        # for i, row in self.df.iterrows():
        #     if row[self.df.columns[0]]==0:
        #         for c,col in enumerate(self.df.columns[1:]): self.df.iloc[i, c+1]=0
        for col in self.df.columns[1:]:
            self.df[col]=self.df.apply(lambda row: 0 if row[self.df.columns[0]]==0 else row[col], axis=1)
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
        if "supressABAReaction" in kwargs:
            if kwargs["supressABAReaction"]==False: kwargs["supressPostABA"]=False
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
        super().__init__(self.cols, df, outlier_threshold=-5, *args, **kwargs, name='Pedestrain')

class VehicleMotion(BaseObject):
    def __init__(self, df, *args, **kwargs):
        self.cols=["BS_v_EgoFAxleLeft_kmh", "BS_v_EgoFAxleRight_kmh", "RDF_val_YawRate"]
        super().__init__(self.cols, df, *args, **kwargs, name='VehicleMotion')

class SingleCSV(object):

    #All Objects that are trackable from Daimler CAN_data csv and are subclasses of BaseObject.
    allObjects=[ABAReaction, MovingObject, StationaryObject, PedestrianObject, VehicleMotion]
    
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
        if  dataObjectsToUse is None:
            dataObjectsToUse=SingleCSV.allObjects
        else: 
            for obj in dataObjectsToUse:
                assert(obj in SingleCSV.allObjects), "dataObjects should be a list of BaseObject subclasses to use"
        #find relevant object to be used to get `edge after ABA reaction` to truncate df
        self.relevantObjectIndex=self.get_relevant_object(df)
        #we find the edgePostABA from the most relevant object
        self.edgePostABA=self.allObjects[self.relevantObjectIndex](df, supressPostABA=False).getEdgePostABA()
        self.kwargs['edgePostABA']=self.edgePostABA
        self.allObjects=[obj(df, **self.kwargs) for obj in dataObjectsToUse]
        self.full_df=df
    def supressCarryForward(self):
        _ = [obj.SupressCarryForward() for obj in self.allObjects]
        return self

    def play(self):
        subprocess.call(["vlc "+vid_from_csv(self.filename)], shell=True)

    @property
    def df(self): #returns Dataframe
        if hasattr(self, "joined_df"): return self.joined_df
        self.joined_df=self.allObjects[0].df.copy()
        for obj in self.allObjects[1:]: self.joined_df=self.joined_df.join(obj.df.copy())
        return self.joined_df
    @property
    def values(self): return self.df.values #numpy equivalent, returns numpy array of dataframe
    @property
    def data(self): return (self.values, self.label) #tuple of (X,y) for dataloader in numpy format
    @property
    def file_id(self): return self.get_file_id(self.filename) #eg: 20170516_015909
    @staticmethod
    def get_file_id(filename): 
        id=filename.split("/")[-1].split(".")[0].split("_")[:3]
        if id[0]=='FLIP': return "_".join(id)
        else: return "_".join(id[:2]) #either FLIP_xxx_xxx or xxx_xxx
    
    @property
    def label(self): # Not hot encoded values. required this way for the learner classes in scripts.learners
        return self.get_label(self.file_id)
    
    @staticmethod
    def get_label(file_id):
        """
        Returns: 
             0 : Left
             1 : Right
             2 : Stop/Other
        """
        path="/home/sufiyan/Common_data/mtp2/dataset/NEW/100_vids/"
        if  file_id in [SingleCSV.get_file_id(filename) for filename in os.listdir(path+"LEFT")]: return 0 #Left Class
        elif file_id in [SingleCSV.get_file_id(filename) for filename in  os.listdir(path+"RIGHT")]: return 1 #Right Class
        else: return 2 #Other class

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
        self.print_relevant_object(self.full_df)
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
    
    def plot(self, i, all_columns=False, **kwargs):
        kwargs={**self.kwargs, **kwargs}
        if all_columns: kwargs["dataObjectsToUse"]=None
        SingleCSV.fromCSV(self.files[i], **kwargs).plot()

    def getSingleCSV(self, i, all_columns=True, **kwargs):
        kwargs={**kwargs, **self.kwargs}
        if all_columns: kwargs["dataObjectsToUse"]=None
        return SingleCSV.fromCSV(self.files[i], **kwargs)

    def play(self, i, **kwargs): SingleCSV.fromCSV(self.files[i], **kwargs).play()

    @classmethod
    def fromCSVFolder(cls, folder, indices=None, **kwargs):
        files=np.asarray([os.path.join(folder, file) for file in os.listdir(folder) if file.split(".")[-1]=='csv'])
        if indices is None: indices = list(range(len(files)))
        return cls(files[indices], **kwargs)


class MovingObjectData(CSVData):
    def __init__(self, files_list, preload=True, **kwargs):
        kwargs["dataObjectsToUse"]=[MovingObject, VehicleMotion] #add dataObject to use rather than all objects
        super().__init__(files_list, preload=preload, **kwargs)
