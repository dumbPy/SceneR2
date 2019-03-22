# from .utils import *
from .utils import StandardSequenceScaler, vid_from_csv
from .core import *

class BaseObject:
    def __init__(self,cols, df, supressOutliers=True, supressPostABA=True, supressCarryForward=False, **kwargs):
        self.df=df.loc[1:,cols]
        self.kwargs=kwargs
        if 'name' in kwargs: self.name=kwargs['name']
        self.cols=cols
        self.SupressOutliers(**self.kwargs)
        if supressPostABA: self.SupressPostABA(**self.kwargs)
        if supressCarryForward: self.SupressCarryForward()
        
    def SupressOutliers(self, outlier_threshold=-50, **kwargs):
        # threshold the folating -200 or -250 values to 0 for better resolution in plotting
        # outlier floating value is different for pedestrian.
        # hence outlier limit of -50 does not work for pedestrian.
        # For pedestrians, when there is no detection, the value goes to around -12.5 
        # that needs to be corrected to 0, with threhold=-5.
        
        for col in self.cols:
            self.df[col]=self.df[col].apply(lambda x: 0 if x< outlier_threshold else x)
        return self
        

    def SupressPostABA(self, edgePostABA=-1, **kwargs):
        """Truncate df when ABA stops tracking relevant object post ABA
        `kwargs['edgePostABA']` is set by SingleCSV.__init__()
        If no edgePostABA is provided, keep whole df
        """
        self.df=self.df.iloc[:edgePostABA,:]
        return self
        
    def SupressCarryForward(self):
        """Supresses the values when 1st column "RDF_typ_ObjType**" is zero. 
        That's when the ABA isn't detecting any object in it's class but the values as carried forward."""
        # for i, row in self.df.iterrows():
        #     if row[self.df.columns[0]]==0:
        #         for c,col in enumerate(self.df.columns[1:]): self.df.iloc[i, c+1]=0
        for col in self.df.columns[1:]:
            self.df[col]=self.df.apply(lambda row: 0 if row[self.df.columns[0]]==0 else row[col], axis=1)
        return self

    def plot(self, ax=None, subplots=True, **kwargs):
        ax=self.df.plot(ax=ax,subplots=True, title=self.name)
        # ax.legend(bbox_to_anchor=(1.5, 1))
        # ax.set_title(self.name)
        return ax
        
class ABAReaction(BaseObject):
    cols=["ABA_typ_WorkFlowState", "OPC_typ_BrakeReq", "ABA_typ_ABAAudioWarn", "ABA_typ_SelObj"]
    def __init__(self, df, cols=None, *args, **kwargs):
        if not cols is None: self.cols=cols
        #Not to supress ABA signals, but supress only CAN signals below
        if "supressABAReaction" in kwargs:
            if kwargs["supressABAReaction"]==False: kwargs["supressPostABA"]=False
        super().__init__(self.cols, df, *args, **kwargs, name='ABA_Reaction')    

class MovingObject(BaseObject):
    cols=["RDF_typ_ObjTypeOr", "RDF_dx_Or", "RDF_v_RelOr", "RDF_dy_Or"]
    def __init__(self, df, cols=None, *args, **kwargs):
        if not cols is None: self.cols=cols
        super().__init__(self.cols, df, *args, **kwargs, name='MovingObj')

class StationaryObject(BaseObject):
    cols=["RDF_typ_ObjTypeOs", "RDF_dx_Os", "RDF_v_RelOs", "RDF_dy_Os"]
    def __init__(self, df, cols=None, *args, **kwargs):
        if not cols is None: self.cols=cols
        super().__init__(self.cols, df, *args, **kwargs, name='StationaryObj')

class PedestrianObject(BaseObject):
    cols=["RDF_typ_ObjTypePed0", "RDF_dx_Ped0", "RDF_vx_RelPed0","RDF_dy_Ped0"]
    def __init__(self, df, cols=None, *args, **kwargs):
        #anything beyond -5 is floating value for pedestrian (assumption based on plots)
        if not cols is None: self.cols=cols
        super().__init__(self.cols, df, outlier_threshold=-5, *args, **kwargs, name='Pedestrain')

class VehicleMotion(BaseObject):
    cols=["BS_v_EgoFAxleLeft_kmh", "BS_v_EgoFAxleRight_kmh", "RDF_val_YawRate"]
    def __init__(self, df, cols=None, *args, **kwargs):
        if not cols is None: self.cols=cols
        super().__init__(self.cols, df, *args, **kwargs, name='VehicleMotion')
        # # diff column
        # self.df["diff"]=self.df["BS_v_EgoFAxleLeft_kmh"]-self.df["BS_v_EgoFAxleRight_kmh"]
        # self.df["diff"]=pd.Series(gaussian_filter1d(self.df["diff"].to_numpy(), sigma=5))
class SingleCSV:

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
                #take out the class from partial for the test
                if isinstance(obj, partial): obj=obj.func
                assert(obj in SingleCSV.allObjects), "dataObjects\
                 should be a list of BaseObject subclasses to use"
        # find relevant object to be used to get `edge after ABA reaction` to truncate df
        self.relevantObjectIndex=self.get_relevant_object(df)
        #we find the edgePostABA from the most relevant object
        #1 will be added to revelantObjectIndex to match SingleCSV.allObjects 
        # rather than Daimler convetion where 0 means Moving Object
        """
        Fix EdgepostABA finding below
        """
        self.edgePostABA=SingleCSV.getEdgePostABA(df, self.relevantObjectIndex, **kwargs)
        self.kwargs['edgePostABA']=self.edgePostABA
        self.allObjects=[obj(df, **self.kwargs) for obj in dataObjectsToUse]
        self.df = df.loc[:,allCols]

    @staticmethod
    def get_rising_edge(column:pd.Series, from_:int=None, to_:int=None, last=True):
        """ Returns index where value rises from `from_` to `to_`
        """
        if      (from_ is None) and (to_ is None):
                condition = lambda val1,val2 : val1<val2
        elif    (from_ is None) and isinstance(to_, int):
                condition = lambda val1,val2 : val1<val2 and val2==to_
        elif    isinstance(from_, int) and (to_ is None):
                condition = lambda val1,val2 : val1<val2 and val1==from_
        elif    isinstance(from_, int) and isinstance(to_, int):
                condition = lambda val1,val2 : val1<val2 and (val1==from_ and val2==to_)
        edges =[i for i,val in enumerate(column[1:], start=column.index[1]) 
                if condition(column[i-1], val)]
        
        if len(edges)>0 :
            if last: return edges[-1]
            else: return edges[0]
        else: return None
    
    @staticmethod
    def get_falling_edge(column:pd.Series, from_:int=None, to_:int=None, last=True):
        """ Returns the index where value falls from `from_` to `to_`
        """
        if      (from_ is None) and (to_ is None):
                condition = lambda val1,val2 : val1>val2
        elif    (from_ is None) and isinstance(to_, int):
                condition = lambda val1,val2 : val1>val2 and val2==to_
        elif    isinstance(from_, int) and (to_ is None):
                condition = lambda val1,val2 : val1>val2 and val1==from_
        elif    isinstance(from_, int) and isinstance(to_, int):
                condition = lambda val1,val2 : val1>val2 and (val1==from_ and val2==to_)
        edges =[i for i,val in enumerate(column[1:], start=column.index[1]) 
                if condition(column[i-1], val)]
        if len(edges)>0 :
            if last: return edges[-1]
            else: return edges[0]
        else: return None

    @staticmethod
    def getEdges(column:pd.Series, threshold=5, **kwargs):
        """returns indices of all edge where laplace gradient is greater than threshold
            Can be used to detect sections of object tracking.
        """
        lap = (laplace(column))
        edges = [idx-1 for i,idx in enumerate(column.index) if abs(lap[i])>threshold]
        return edges # return idx just before |laplace(x)| > threshold

    @staticmethod
    def getABAReactionIndex(df):
        # Old method, here incase I want to revert to it
        # if len(df[df["ABA_typ_WorkFlowState"]>0]["ABA_typ_WorkFlowState"].index) < 2 : return df.index[-1]
        index = SingleCSV.get_rising_edge(df["ABA_typ_WorkFlowState"], from_=0, last=True)
        assert(index is not None), "No ABA Reaction Found in a File"
        return index

    @staticmethod
    def getABAReactionStopIndex(df):
        """Return the index where ABA stops reacting.
        i.e., When ABA_typ_WorkFlowState falls to 0 or rises to 5(full braking stop)
        
        If ABA reacts till last frame, Return Last Index
        """
        reaction_index = SingleCSV.getABAReactionIndex(df)
        edge_to_0 = SingleCSV.get_falling_edge(df["ABA_typ_WorkFlowState"][reaction_index:], to_=0, last=True)
        # ABA reaction 4 means Full Braking and 5 means Full braking Ended.
        edge_to_5 = SingleCSV.get_rising_edge(df["ABA_typ_WorkFlowState"][reaction_index:], to_=5, last=True)
        edges = [edge_to_0, edge_to_5]
        edges = sorted(filter(None, edges)) # Filter out None values
        if len(edges)==0 : return df.index[-1]
        return edges[0] # First time ABA stops after last ABA reaction

    @staticmethod
    def getEdgePostABA(df, relevantObjectIndex:int, verbose=False, **kwargs):
        """
        Args:
        -----
        df :                    pandas DataFrame of CAN signal csv
        relevantObjectIndex :   relevant Object that caused ABA reaction.
                                0 : Moving Object
                                1 : Stationary Object
                                2 : Pedestrian A
                                3 : Pedestrian B
        threshold :             threshold for laplace gradient. checkout `SingleCSV.getEdges`

        Returns
        --------
        edgePostABA :           Edge where relevant object's tracking stops
                                Or Edge where ABA stops braking/audio warning
                                i.e., `ABA_typ_WorkFlowState`'s falling edge
        """
        # take 4 columns of ABAReactions and columns of relevantObject
        # Use relevantObjectIndex+1 as SingleCSV.allObjects starts with ABAReaction not MovingObject
        df=df.loc[:, ABAReaction.cols+SingleCSV.allObjects
                [relevantObjectIndex+1].cols]
        
        ABAReactionIndex=SingleCSV.getABAReactionIndex(df)
        # Edge where ABA reaction stops. 50 is added as vehicle action
        # happens usually after ABA reaction For eg. ABA might apply
        # break and stop, and then the vehicle ahead might turn (usually within 1 sec)
        ABAReactionStopIndex=SingleCSV.getABAReactionStopIndex(df)+50
        # use relevant objects's 0th column representing object tracking
        # to find edge eg. `RDF_typ_ObjTypeOr`
        edges_0 =   [SingleCSV.get_falling_edge(
            df[SingleCSV.allObjects[relevantObjectIndex+1]
            .cols[0]][ABAReactionIndex:ABAReactionStopIndex], last=False)]
        # Filter out None Value that get_falling_edge might return in case it
        # didnt find the edge it was loking for
        edges_0 =   [i for i in filter(None, edges_0)]
        
        # Sometimes, Radar switches from one vehicle to another without `RDF_typ_ObjTypeOr` falling
        # If 'RDF_dx_Or' changes abruptly, use its location as edge 
        edges_1 = SingleCSV.getEdges(df[SingleCSV.allObjects
        [relevantObjectIndex+1].cols[1]]
        [ABAReactionIndex:ABAReactionStopIndex], **kwargs)
        if verbose:
            print(edges_0)
            print("ABAReactionIndex: ", ABAReactionIndex)
            print("ABAReactionStopIndex: %i"%ABAReactionStopIndex)
            print("Edge_0: ",edges_0)
            print("Edge_1: ",edges_1)
        supressable_edges = sorted(edges_0+edges_1+[ABAReactionStopIndex])
        if len(supressable_edges)>0 : return supressable_edges[0]
        else:
            edge = df[df["ABA_typ_WorkFlowState"]>0]["ABA_typ_WorkFlowState"].index[-1]
        # return value just before it abruptly changed, hence `edge-1`
        return edge-1
        
    def supressCarryForward(self):
        _ = [obj.SupressCarryForward() for obj in self.allObjects]
        return self

    def play(self, player = None, **kwargs):
        print("Name: ",__name__)
        from IPython.display import Video, HTML
        if player:
            subprocess.call([player+
                " "+vid_from_csv(self.file_id)], shell=True)
            return
        try: # if in notebook, this loop runs and Video object is returned
            get_ipython
            return Video(vid_from_csv(self.file_id), embed=True)
        # If running from terminal, `get_ipython` will raise error 
        # and `globalVariable.video_player` will be called
        except: subprocess.call([globalVariables.video_player+
                " "+vid_from_csv(self.file_id)], shell=True)

    @property
    def full_df(self): #returns full Dataframe
            return read_csv_auto(self.filename)
        
    @property
    def dy(self):
        """Returns orthogonal distance column of relevant object
        For moving object, returns RDF_dy_Or,
        For Stationary object, returns RDF_dy_Os
        For Pedestrian, returns, RDF_dy_Ped0
        """
        dy_col = [col for col in  SingleCSV.allObjects
                 [self.relevantObjectIndex+1].cols if 'dy' in col]
        return self.df[dy_col]
    @property
    def values(self): return self.df.values #numpy equivalent, returns numpy array of dataframe
    @property
    def data(self): return (self.values, self.label) #tuple of (X,y) for dataloader in numpy format
    @property
    def file_id(self): return self.get_file_id(self.filename) #eg: 20170516_015909
    @staticmethod
    def get_file_id(filename): 
        id=filename.split(os.sep)[-1].split(".")[0].split("_")[:3]
        if id[0]=='FLIP': return "_".join(id)
        else: return "_".join(id[:2]) #either FLIP_xxx_xxx or xxx_xxx
    
    @property
    def label(self): # Not hot encoded values. required this way for the learner classes in SceneR2.learners
        return self.get_label(self.file_id)
    
    @staticmethod
    def get_label(file_id):
        """
        Returns: 
             0 : Left
             1 : Right
             2 : Stop/Other
        """
        # path="/home/sufiyan/Common_data/mtp2/dataset/NEW/100_vids/"
        # if  file_id in [SingleCSV.get_file_id(filename) for filename in os.listdir(path+"LEFT")]: return 0 #Left Class
        # elif file_id in [SingleCSV.get_file_id(filename) for filename in  os.listdir(path+"RIGHT")]: return 1 #Right Class
        # else: return 2 #Other class
        with open (globalVariables.path_to_pickled_labels, 'rb') as f:
            left_labels,right_labels=pickle.load(f)
        if file_id in left_labels: return 0
        elif file_id in right_labels: return 1
        else: return 2

    @classmethod
    def fromCSV(cls, filename, **kwargs):
        return cls(read_csv_auto(filename), filename=filename, **kwargs)
    
    # As defined in `DML Signal List for Image Analysis Study.xlsx shared by Dr Tilak`
    relevantObjects={0:"Driving/Moving Object", 1:"Stationary Object", 2:"Pedestrian A", 3:"Pedestrian B"}

    @staticmethod
    def get_relevant_object(df:pd.DataFrame):
        """
        Args:
        ------
        df : pandas dataframe of CAN signal from csv
        
        Returns
        --------
        Index of object that caused ABA reaction as from `ABA_typ_SelObj` colums
        relevantObjectIndex:  
                0 : Moving Object
                1 : Stationary Object
                2 : Pedestrian A
                3 : Pedestrian B
        returns index as per Daimlers convention.
        rather than `SingleCSV.allObjects`"""
        ABA_ReactionIndex = SingleCSV.getABAReactionIndex(df)
        relevantObjectIndex=df["ABA_typ_SelObj"][ABA_ReactionIndex]
        return relevantObjectIndex

    @staticmethod
    def print_relevant_object(df):
        relevantObjectIndex=SingleCSV.get_relevant_object(df)
        print("Reason for Braking: ", SingleCSV.relevantObjects[relevantObjectIndex])

    def plot(self, verbose=True, **kwargs):
        if verbose:
            self.print_relevant_object(self.full_df)
            print("Label: ", self.label)
            print("edgePostABA: ", self.edgePostABA)
        kwargs = {'verbose':verbose, **kwargs}
        for obj in self.allObjects: obj.plot(**kwargs)

    def show_as_image(self, ax=None):
        if ax==None: f,ax=plt.subplots(1,1)
        ax.imshow(self.values, extent=[0,10,0,10])
        plt.show()
        return ax

class CSVData(data.Dataset):
    def __init__(self, files_list, preload=None, skip_labels=[], **kwargs):
        """
        Inputs
        ------
        files_list:         list of CAN signal csv paths
        preload:            To preload the arrays  of each file to train faster (Now Deprecated)
                            If False, will load 100 files to calculate mean and std of each column
        skip_labels:        list if labels for csv to skip. eg, skip_label=[2] to skip all 
                            files with pedestrian crossing as the reason for braking
                            0 : Moving object
                            1 : Stationary Object
                            2 : Pedestrian A is the reason for ABA reaction 
                            3 : Pedestrian B is the reason for ABA reaction 

        """
        self.files=files_list
        self.kwargs=kwargs
        super().__init__()
        self.standardScaler=StandardSequenceScaler() #scales data to 0 mean and unit variance
        self.preload=preload
        if preload:
            import warnings
            warnings.warn("preload is deprecated. Now all files are preloaded for `skip_labels` to work")
        self.kwargs['skip_labels']=skip_labels
        self.files=[filename for filename in self.files 
                    if not SingleCSV.get_relevant_object(read_csv_auto(filename)) 
                    in skip_labels]
        self.data = [SingleCSV.fromCSV(filename, **self.kwargs).data for i,filename in enumerate(self.files)]
        self.standardScaler.fit([x for x,y in self.data])
        self.data = [(self.standardScaler.transform(x),y) for x,y in self.data]
        # else:
        #     import warnings
        #     warnings.warn("preload=False will load first 100 CSVs to calculate mean and std for standardScalar")
        #     self.data = [SingleCSV.fromCSV(self.files[i], **self.kwargs).data for i in range(min(100, self.__len__()))]
        #     self.standardScaler.fit([x for x,y in self.data])

    def __len__(self): return len(self.data)
    
    def __getitem__(self, i):
        return self.data[i]
        # else: 
        #     x,y=SingleCSV.fromCSV(self.files[i], **self.kwargs).data
        #     return (self.standardScaler.transform(x) ,y)
    
    def plot(self, i, supressPostABA=True, all_columns=False, **kwargs):
        kwargs={**self.kwargs, **kwargs, 'supressPostABA':supressPostABA}
        if all_columns: kwargs["dataObjectsToUse"]=None
        SingleCSV.fromCSV(self.files[i], **kwargs).plot(**kwargs)

    def getSingleCSV(self, i, all_columns=True, **kwargs):
        kwargs={**kwargs, **self.kwargs}
        if all_columns: kwargs["dataObjectsToUse"]=None
        return SingleCSV.fromCSV(self.files[i], **kwargs)

    def play(self, i, player=None, **kwargs) : return  SingleCSV.fromCSV(self.files[i], **kwargs).play(player=player,**kwargs)

    @classmethod
    def fromCSVFolder(cls, folder:str, indices=None, skip_labels=[], **kwargs):
        """
        Arguments
        ------
        folder:         path to folder that has all CAN signal csv to load
        indices:        indices of the csvs to load. two set of mutually exclusive indices 
                        for train and test can be used
        skip_labels:    list if labels for csv to skip. eg, skip_label=[2] to skip all files with
                        pedestrian crossing as the reason for braking
                        0 : Moving object
                        1 : Stationary Object
                        2 : Pedestrian A is the reason for ABA reaction 
                        3 : Pedestrian B is the reason for ABA reaction 

        """
        files=np.asarray([os.path.join(folder, file) for file in os.listdir(folder) 
                                                     if file.split(".")[-1]=='csv'])
        if indices is None: indices = list(range(len(files)))
        return cls(files[indices], skip_labels=skip_labels, **kwargs)

class MovingObjectData(CSVData):
    def __init__(self, files_list, **kwargs):
        vhMotion=partial(VehicleMotion, cols=["RDF_val_YawRate"])
        kwargs["dataObjectsToUse"]=[MovingObject, vhMotion] #add dataObject to use rather than all objects
        super().__init__(files_list, **kwargs)

class MovingObjectData2(CSVData):
    """MovingObjectData2 has only 2 columns, 1 for y position of moving object, and 2nd for yaw rate"""
    def __init__(self, files_list, **kwargs):
        mvObj=partial(MovingObject, cols=["RDF_dy_Or"])
        # vhMotion=partial(VehicleMotion, cols=["RDF_val_YawRate"])
        kwargs["dataObjectsToUse"]=[mvObj, VehicleMotion] #add dataObject to use rather than all objects
        super().__init__(files_list, **kwargs)

    