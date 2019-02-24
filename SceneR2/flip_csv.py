if __name__=='__main__':
    from core import *
    from dataset import *
else:
    from .core import *
    from .dataset import *

path= "/home/sufiyan/data/Daimler/100_vids/csv/"
filenames=[path+filename for filename in os.listdir(path) if filename.split('.')[-1]=='csv']
print(filenames)

filenames=[filename for filename in filenames if SingleCSV.get_label(SingleCSV.get_file_id(filename)) in [0,1]]
y_columns=[col for col in allCols if col.split('_')[1]=='dy']
print(y_columns)
print(filenames)
for filename in filenames:
    df=read_csv_auto(filename)
    for col in y_columns:
        df[col] = df[col].apply(lambda value: -value)
    df.to_csv(addFlip(filename), index=False)
    print(filename)
    
def addFlip(filename):
    filename=filename.split("/")
    filename[-1]="FLIP_"+filename[-1]
    filename="/".join(filename)
    return filename
    