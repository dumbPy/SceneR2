if not __name__=="__main__":
    from .core import *
else:
    from core import *

allCols=["ABA_typ_WorkFlowState OPC_typ_BrakeReq ABA_typ_ABAAudioWarn ABA_typ_SelObj BS_v_EgoFAxleLeft_kmh BS_v_EgoFAxleRight_kmh RDF_val_YawRate RDF_typ_ObjTypeOr RDF_dx_Or RDF_v_RelOr RDF_dy_Or RDF_typ_ObjTypeOs RDF_dx_Os RDF_v_RelOs RDF_dy_Os RDF_typ_ObjTypePed0 RDF_dx_Ped0 RDF_vx_RelPed0 RDF_dy_Ped0 "]
allCols=allCols[0].split()

class BaseObject:
    def __init__(self,cols, df, supressOutliers=True, supressPostABA=True, **kwargs):
        self.data=df.loc[1:,cols]
        self.kwargs=kwargs
        if 'name' in kwargs: self.name=kwargs['name']
        self.cols=cols
        self.df=df.loc[:,allCols]
        self.SupressOutliers(**self.kwargs)
        if supressPostABA: self.SupressPostABA()
        
    def SupressOutliers(self, outlier_limit=-50, **kwargs):
        #threshold the folating -200 or -250 values to 0 for better resolution in plotting
            #outlier floating value is different for pedestrian.
            #hence outlier limit of -50 does not work for pedestrian.
            #For pedestrians, when there is no detection, the value goes to around -12.5 
            #that needs to be corrected to 0.
        try: outlier_limit=self.kwargs["outlier_limit"]
        except:pass
        for col in self.cols:
            self.data[col]=self.data[col].apply(lambda x: 0 if x< outlier_limit else x)
        return self
        
    def SupressPostABA(self, countBeyondABA=100, **kwargs):
        index=self.df[self.df["ABA_typ_WorkFlowState"]>0]["ABA_typ_WorkFlowState"].index[1]
        self.data=self.data.iloc[:index+countBeyondABA,:]
        return self
        
    def SupressCarryForward(self):
        """Supresses the values when 1st column "RDF_typ_ObjType**" is zero. 
        Then's when the ABA isn't detecting any aboect in it's class but the values as carried forward."""
        for i, row in self.data.iterrows():
            if row[self.data.columns[0]]==0:
                for c,col in enumerate(self.data.columns[1:]): self.data.iloc[i, c+1]=0
        return self
    
    def plot(self, ax=None, **kwargs):
        ax=self.data.plot(ax=ax,subplots=True, title=self.name, **kwargs)
#         ax.legend(bbox_to_anchor=(1.5, 1))
#         ax.set_title(self.name)
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

class SingleCSV(nn.Module):
    def __init__(self, df):
        self.abaReaction=ABAReaction(df)
        self.movingObj=MovingObject(df)
        self.stationaryObj=StationaryObject(df)
        self.pedestrian=PedestrianObject(df)
        self.allObjects=[self.abaReaction, self.movingObj, self.stationaryObj, self.pedestrian]

    def getData(self):
        if hasattr(self, data): return self.data
        else: self.data=self.abaReaction.data.copy()
        for obj in self.allObjects: self.data.join(obj.data)
        return self.data

    @classmethod
    def fromCSV(cls, fileName):
        try:
            df=pd.read_csv(fileName)
            if not "ABA_typ_WorkFlowState" in df.columns: raise AttributeError
        except: df=pd.read_csv(fileName, delimiter=';')
        return cls(df)

    @staticmethod
    def readCSV(name):
        try:
            df=pd.read_csv(name)
            if not "ABA_typ_WorkFlowState" in df.columns: raise AttributeError
        except: df=pd.read_csv(name, delimiter=';')
        return df


class CSVData(nn.Module):
    def __init__(self, files_list, **kwargs):
        self.files=files_list
        self.kwargs=kwargs
    
    def __len__(self): return len(self.files)
    
    def __getitem__(self, i):
        return SingleCSV.fromCSV(self.files[i], **self.kwargs).getData()

    @classmethod
    def fromCSVFolder(cls, folder, **kwargs):
        files=[os.path.join(file) for file in os.listdir(folder) if file.split(".")[-1]=='csv']
        return cls(files, **kwargs)



