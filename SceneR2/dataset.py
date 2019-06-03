from .utils.dataset import StandardSequenceScaler, vid_from_csv, VidLoader, makeEven
from .core import *
from .errors import NoMovingRelevantObjectInData

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
        `kwargs['edgePostABA']` is set by SingleCAN.__init__()
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

    def plot(self, edgePostABA=None, tight_layout=False, **kwargs):
        """Plot the object's columns. pass edgePostABA to plot
        tight_layout = True for removing white spaces 
                            (implemented for saving image for SceneR2WebApp slider)
        figsize = (5,3) for good sized gifs and resizing the fig for less
        number of columns (see code below)
        """
        # If figsize provided in kwargs (generally for 4 columns)
        # If less than 4 columns in object, reduce figsize height
        if 'figsize' in kwargs:
            figsize = kwargs['figsize']
        else: figsize=(5,3)
        w,h = figsize
        h = (h/4)*len(self.df.columns)
        figsize = (w,h)
        
        axes=self.df.plot(subplots=True, title=self.name, figsize=figsize)
        # Draw the edgePostABA verticle line on plots
        if edgePostABA:
            if edgePostABA<=self.df.index[-1]:
                for i,ax in enumerate(axes.flat):
                    ax.axvline(edgePostABA, color='#4E4F4C')
                    ax.fill_between([0, edgePostABA], *ax.get_ylim(), facecolor='grey', alpha=0.2)

        
        fig = axes.flat[0].get_figure()
        # for ax in axes.flat: ax.set_axis_off() # turn off the axis markers
        # for ax in axes.flat: ax.yaxis.tick_right()
        
        if tight_layout:
            fig.subplots_adjust(0,0,1,0.9,0,0.1) # remove whitespaces from sides
            for ax in axes.flat: ax.tick_params(axis='y', direction='in', pad=-40) # turn off the axis markers
            fig.canvas.draw() # draw the figure but don't show
        return axes
        
class TrackingObject(BaseObject):
    def __init__(self,*args, **kwargs):
        """check BaseObject init for arguments"""
        super().__init__(*args, **kwargs)
        self.clip_y()

    def add_vid_detection_state(self, state:'binary list')-> None:
        """Given state binary list, this method extends it to have double fps for can and copies last value untill length equal to can"""
        assert(len(state)<=self.df.shape[0]/2), f"video detection state length {len(state)} should be less than hald of can length {self.df.shape[0]}"
        state=[state[i] if i%2==0 else state[i-1] for i in len(state)]
        while len(state)<self.df.shape[0]:state.append(state[-1])
        self.df.insert(1, self.__class__.__name__+'_Det_from_vid', state)

    @property
    def y(self):
        dy_cols = [col for col in self.cols if '_dy_' in col]
        assert(len(dy_cols)==1),f'No dy col found in {self.__class__.__name__}'
        return self.df[dy_cols[0]]

    def clip_y(self):
        dy_col = [col for col in self.cols if '_dy_' in col][0]
        self.df[dy_col] = self.df[dy_col].clip(-3.9, 3.9)

    @property
    def det_col(self):
        det_col = [col for col in self.cols if '_typ_' in col]
        assert(len(det_col)==1),f"detection column cannot be returned when {self.__class__.__name__} has no RDF_typ_ObjTypexx col"
        return self.df[det_col[0]]


class ABAReaction(BaseObject):
    cols=["ABA_typ_WorkFlowState", "OPC_typ_BrakeReq", "ABA_typ_ABAAudioWarn", "ABA_typ_SelObj"]
    def __init__(self, df, cols=None, *args, **kwargs):
        if not cols is None: self.cols=cols
        #Not to supress ABA signals, but supress only CAN signals below
        if "supressABAReaction" in kwargs:
            if kwargs["supressABAReaction"]==False: kwargs["supressPostABA"]=False
        super().__init__(self.cols, df, *args, **kwargs, name='ABA_Reaction')    

class MovingObject(TrackingObject):
    cols=["RDF_typ_ObjTypeOr", "RDF_dx_Or", "RDF_v_RelOr", "RDF_dy_Or"]
    def __init__(self, df, cols=None, *args, **kwargs):
        if not cols is None: self.cols=cols
        super().__init__(self.cols, df, *args, **kwargs, name='MovingObj')

class StationaryObject(TrackingObject):
    cols=["RDF_typ_ObjTypeOs", "RDF_dx_Os", "RDF_v_RelOs", "RDF_dy_Os"]
    def __init__(self, df, cols=None, *args, **kwargs):
        if not cols is None: self.cols=cols
        super().__init__(self.cols, df, *args, **kwargs, name='StationaryObj')

class PedestrianObject(TrackingObject):
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
class SingleCAN:

    #All Objects that are trackable from Daimler CAN_data csv and are subclasses of BaseObject.
    allObjects=[ABAReaction, MovingObject, StationaryObject, PedestrianObject, VehicleMotion]
    
    def __init__(self, df:pd.DataFrame, filename, label=None, dataObjectsToUse:list=None, **kwargs):
        """
        df: Pandas dataFrame to be cleaned and parsed
        dataObjetsToUse: list of BaseObject subclasses to use to parse the dataFrame.
                         dafault [ABAReaction, MovingObject, StationaryObject, PedestrianObject]
        these dataObjectsToUse will be initialized (and internally cleaned) 
        and joined again to return with SingleCAN.df
        """
        
        self.kwargs=kwargs
        self.filename=filename
        # for test videos that are unseen, give label or it will give error
        if label is not None: self._label = label
        if  dataObjectsToUse is None:
            dataObjectsToUse=SingleCAN.allObjects
        else: 
            for obj in dataObjectsToUse:
                #take out the class from partial for the test
                if isinstance(obj, partial): obj=obj.func
                assert(obj in SingleCAN.allObjects), "dataObjects\
                 should be a list of BaseObject subclasses to use"
        # find relevant object to be used to get `edge after ABA reaction` to truncate df
        self.relevantObjectIndex=self.get_relevant_object(df)
        #we find the edgePostABA from the most relevant object
        #1 will be added to revelantObjectIndex to match SingleCAN.allObjects 
        # rather than Daimler convetion where 0 means Moving Object
        """
        Fix EdgepostABA finding below
        """
        self.edgePostABA=SingleCAN.getEdgePostABA(df, self.relevantObjectIndex, **kwargs)
        self.kwargs['edgePostABA']=self.edgePostABA
        self.allObjects=[obj(df, **self.kwargs) for obj in dataObjectsToUse]
        # self.df = df.loc[:,allCols]
        self.df = pd.concat([obj.df for obj in self.allObjects], axis=1)
        # relevantObject is initialized to be able to extract dy column from 
        # it, independent of the self.allObjects which might or might not 
        # contain dy column of the relevantObject, as it depends on the 
        # dataObjectsToUse argument, that might have select few columns, eg- 
        # dataObjectsToUse = [partial(MovingObject, cols=[RDF_dx_Or])]
        # and might completely skip the RDF_dy_Or column in the dataset
        self.relevantObject = \
                SingleCAN.allObjects[self.relevantObjectIndex+1](df)
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
        index = SingleCAN.get_rising_edge(df["ABA_typ_WorkFlowState"], from_=0, last=True)
        assert(index is not None), "No ABA Reaction Found in a File"
        return index

    @staticmethod
    def getABAReactionStopIndex(df):
        """Return the index where ABA stops reacting.
        i.e., When ABA_typ_WorkFlowState falls to 0 or rises to 5(full braking stop)
        
        If ABA reacts till last frame, Return Last Index
        """
        reaction_index = SingleCAN.getABAReactionIndex(df)
        edge_to_0 = SingleCAN.get_falling_edge(df["ABA_typ_WorkFlowState"][reaction_index:], to_=0, last=True)
        # ABA reaction 4 means Full Braking and 5 means Full braking Ended.
        edge_to_5 = SingleCAN.get_rising_edge(df["ABA_typ_WorkFlowState"][reaction_index:], to_=5, last=True)
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
        threshold :             threshold for laplace gradient. checkout `SingleCAN.getEdges`

        Returns
        --------
        edgePostABA :           Edge where relevant object's tracking stops
                                Or Edge where ABA stops braking/audio warning
                                i.e., `ABA_typ_WorkFlowState`'s falling edge
        """
        # take 4 columns of ABAReactions and columns of relevantObject
        # Use relevantObjectIndex+1 as SingleCAN.allObjects starts with ABAReaction not MovingObject
        df=df.loc[:, ABAReaction.cols+SingleCAN.allObjects
                [relevantObjectIndex+1].cols]
        
        ABAReactionIndex=SingleCAN.getABAReactionIndex(df)
        # Edge where ABA reaction stops. 50 is added as vehicle action
        # happens usually after ABA reaction For eg. ABA might apply
        # break and stop, and then the vehicle ahead might turn (usually within 1 sec)
        ABAReactionStopIndex=SingleCAN.getABAReactionStopIndex(df)+50
        # use relevant objects's 0th column representing object tracking
        # to find edge eg. `RDF_typ_ObjTypeOr`
        edges_0 =   [SingleCAN.get_falling_edge(
            df[SingleCAN.allObjects[relevantObjectIndex+1]
            .cols[0]][ABAReactionIndex:ABAReactionStopIndex], last=False)]
        # Filter out None Value that get_falling_edge might return in case it
        # didnt find the edge it was loking for
        edges_0 =   [i for i in filter(None, edges_0)]
        
        # Sometimes, Radar switches from one vehicle to another without `RDF_typ_ObjTypeOr` falling
        # If 'RDF_dx_Or' changes abruptly, use its location as edge 
        edges_1 = SingleCAN.getEdges(df[SingleCAN.allObjects
        [relevantObjectIndex+1].cols[1]]
        [ABAReactionIndex:ABAReactionStopIndex], **kwargs)
        if verbose:
            print(edges_0)
            print("ABAReactionIndex: ", ABAReactionIndex)
            print("ABAReactionStopIndex: %i"%ABAReactionStopIndex)
            print("Edge_0: ",edges_0)
            print("Edge_1: ",edges_1)
        supressable_edges = sorted(edges_0 + edges_1 + [ABAReactionStopIndex])
        if len(supressable_edges)>0 :
            return makeEven(supressable_edges[0])
        else:
            edge = df[df["ABA_typ_WorkFlowState"]>0]["ABA_typ_WorkFlowState"].index[-1]
            # return value just before it abruptly changed, hence `edge-1`
            return makeEven(edge-1)
        
    def supressCarryForward(self):
        _ = [obj.SupressCarryForward() for obj in self.allObjects]
        return self
    
    @property
    def vid_file(self): return vid_from_csv(self.file_id)
    @property
    def vid_loader(self): 
        """Returns the `SceneR2.utils.dataset.VidLoader` object
        edgePostABA is divided by 2 as framerate (25 fps) is half the 
        frequency (100 hz) of CAN, i.e., 1000 readings in 20 sec"""

        return VidLoader(self.vid_file, int(self.edgePostABA/2))

    def play(self, player = None, **kwargs):
        from IPython.display import Video, HTML
        if player:
            subprocess.call([player+
                " "+self.vid_file], shell=True)
            return
        try: # if in notebook, this loop runs and Video object is returned
            get_ipython
            return Video(self.vid_file, embed=True)
        # If running from terminal, `get_ipython` will raise error 
        # and `globalVariable.video_player` will be called
        except: subprocess.call([globalVariables.video_player+
                " "+self.vid_file], shell=True)

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
        dy_col = [col for col in self.relevantObject.cols if 'dy' in col][0]
        return self.relevantObject.df[dy_col]
    @property
    def values(self): return self.df.values #numpy equivalent, returns numpy array of dataframe
    @property
    def data(self): return (self.values[:self.edgePostABA], self.label) #tuple of (X,y) for dataloader in numpy format
    @property
    def file_id(self): return self.get_file_id(self.filename) #eg: 20170516_015909
    @staticmethod
    def get_file_id(filename): 
        id=filename.split(os.sep)[-1].split(".")[0].split("_")[:3]
        if id[0]=='FLIP': return "_".join(id)
        else: return "_".join(id[:2]) #either FLIP_xxx_xxx or xxx_xxx
    
    @property
    def label(self): # Not hot encoded values. required this way for the learner classes in SceneR2.learners
        if hasattr(self, '_label'): return self._label
        else: self._label = self.get_label(self.file_id)
        return self._label
    
    @staticmethod
    def get_label(file_id):
        """
        Returns: 
             0 : Left
             1 : Right
             2 : Stop/Other
        """
        
        with open (globalVariables.path_to_pickled_labels, 'rb') as f:
            left_labels,right_labels=pickle.load(f)
        if file_id in left_labels: return 0
        elif file_id in right_labels: return 1
        else: return 2

    @classmethod
    def fromCSV(cls, filename, **kwargs):
        return cls(read_csv_auto(filename), filename=filename, **kwargs)
    
    # As defined in `DML Signal List for Image Analysis Study.xlsx shared by Dr Tilak`
    relevantObjectsDict={0:"Driving/Moving Object", 1:"Stationary Object", 2:"Pedestrian A", 3:"Pedestrian B"}

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
        rather than `SingleCAN.allObjects`"""
        ABA_ReactionIndex = SingleCAN.getABAReactionIndex(df)
        relevantObjectIndex=df["ABA_typ_SelObj"][ABA_ReactionIndex]
        return relevantObjectIndex

    def add_vid_detections(self, detections):
        # TODO implement this method
        """Yet to be implemented"""
        pass
    
    @staticmethod
    def print_relevant_object(df):
        relevantObjectIndex=SingleCAN.get_relevant_object(df)
        print("Reason for Braking: ", SingleCAN.relevantObjectsDict[relevantObjectIndex])

    def plot(self, verbose=True, **kwargs):
        if verbose:
            self.print_relevant_object(self.full_df)
            print("Label: ", self.label)
            print("edgePostABA: ", self.edgePostABA)
        kwargs = {'verbose':verbose, **kwargs}
        all_axes = []
        for i,obj in enumerate(self.allObjects): 
            ax =  obj.plot(edgePostABA=self.edgePostABA, **kwargs)
            all_axes.append(ax)
        return all_axes

    def show_as_image(self, ax=None):
        if ax==None: f,ax=plt.subplots(1,1)
        ax.imshow(self.values, extent=[0,10,0,10])
        plt.show()
        return ax

class CANData(data.Dataset):
    def __init__(self, can_list, skip_labels=[], return_video=False, force=False, **kwargs):
        """
        Inputs
        ------
        can_list:            list of CAN signal csv paths
                            If False, will load 100 files to calculate mean and std of each column
        skip_labels:        list if labels for csv to skip. eg, skip_label=[2] to skip all 
                            files with pedestrian crossing as the reason for braking
                            0 : Moving object
                            1 : Stationary Object
                            2 : Pedestrian A is the reason for ABA reaction 
                            3 : Pedestrian B is the reason for ABA reaction 
        force:              Force the dataset to add the can data 
                            irrespective of the relevant object from skip_labels args. Used  mainly for plotting if webapp

        """
        self._return_video = return_video
        super().__init__()
        self.standardScaler=StandardSequenceScaler() #scales data to 0 mean and unit variance
        self.kwargs=kwargs
        self.kwargs['skip_labels']=skip_labels

        self.data = []
        self.vid_loaders = []
        self.can  = []

        for _,can_file in enumerate(tqdm(can_list)):
            c = SingleCAN.fromCSV(can_file, **self.kwargs)
            # If not for process, add can data anyhow for plotting
            if force: self.can.append(c.filename)
            if not c.relevantObjectIndex in skip_labels and not force:
                self.can.append(c.filename)
                self.data.append(c.data)
                if return_video: self.vid_loaders.append(c.vid_loader)
        # self.data = [SingleCAN.fromCSV(filename, **self.kwargs).data for i,filename in enumerate(self.can)]
        if not force:
            if len(self.data)==0: raise NoMovingRelevantObjectInData
            # process loads the numpy array data of all CANs and transforms them and is then converted to tensors in __getitem__ method
            self.standardScaler.fit([x for x,y in self.data])
            self.data = [(self.standardScaler.transform(x),y) for x,y in self.data]

    @property
    def return_video(self): return self._return_video
    @return_video.setter
    def return_video(self, new_state:bool): self._return_video=new_state
    @staticmethod
    def sync_can(can:np.array):
        """Sync the number of frames in can and video
        i.e., only return even indexs of can data"""
        return can[[i%2!=0 for i in range(len(can))]]

    def __len__(self): return len(self.data)
    
    def __getitem__(self, i):
        """
        If self._return_video: return ((Video, CAN), label)
        Else return (CAN, label)
        """
        if self.return_video:
            vid = self.vid_loaders[i].data
            can = self.sync_can(self.data[i][0])
            # Edge case Handled below. When ABA Reaction is at the very end.
            # edgePostABA=1000, can size 999, sync_can returns size (999-1)/2 = 499, video size 498
            can = can[:len(vid)]
            return ((vid, can), self.data[i][1])
        else: return self.data[i]
        # else: 
        #     x,y=SingleCAN.fromCSV(self.can[i], **self.kwargs).data
        #     return (self.standardScaler.transform(x) ,y)
    
    def plot(self, i, supressPostABA=True, all_columns=False, **kwargs):
        kwargs={**self.kwargs, **kwargs, 'supressPostABA':supressPostABA}
        if all_columns: kwargs["dataObjectsToUse"]=None
        return SingleCAN.fromCSV(self.can[i], **kwargs).plot(**kwargs)

    def getSingleCAN(self, i, all_columns=True, **kwargs):
        kwargs={**kwargs, **self.kwargs}
        if all_columns: kwargs["dataObjectsToUse"]=None
        return SingleCAN.fromCSV(self.can[i], **kwargs)

    def play(self, i, player=None, **kwargs) : 
        return  SingleCAN.fromCSV(self.can[i], **kwargs).play(player=player,**kwargs)

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
        can = np.asarray(glob(os.path.join(folder, '*.csv')))
        if indices is None: indices = list(range(len(can)))
        return cls(can[indices], skip_labels=skip_labels, **kwargs)
    
class MovingObjectData(CANData):
    # add dataObject to
    # use rather than all objects
    vhMotion=partial(VehicleMotion, cols=["RDF_val_YawRate"])
    kwargs={}
    kwargs["dataObjectsToUse"]=[MovingObject, vhMotion]

    def __init__(self, can_list, **kwargs):
        kwargs['skip_labels']=[2,3]
        super().__init__(can_list, **kwargs, **self.kwargs)

class MovingObjectData2(CANData):
    """MovingObjectData2 has only 4 columns, 1 for y position of moving object, and 3 for vehicle position"""

    mvObj=partial(MovingObject, cols=["RDF_dy_Or"])
    # vhMotion=partial(VehicleMotion, cols=["RDF_val_YawRate"])
    kwargs={}
    kwargs["dataObjectsToUse"]=[mvObj, VehicleMotion] #add dataObject to use rather than all objects

    def __init__(self, can_list, **kwargs):
        kwargs['skip_labels']=[2,3]
        super().__init__(can_list, **kwargs, **self.kwargs)

class FusionData(MovingObjectData2):
    def __init__(self, can_list, skip_labels=[2,3], **kwargs):
        assert('return_video' not in kwargs),"Use CANData if you don't want videos"
        super().__init__(can_list, skip_labels=skip_labels, return_video=True, **kwargs)
