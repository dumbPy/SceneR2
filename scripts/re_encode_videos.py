#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'Notebooks'))
	print(os.getcwd())
except:
	pass

#%%
import os, subprocess, numpy as np, tqdm


#%%
PATH="/home/sufiyan/Common_data/mtp2/dataset/OLD"


#%%
def change_to_new(old_path):
    vid = old_path.split("/")[-1]
    vid=vid.split(".")[0]+".mp4"
    path="/home/sufiyan/data/Daimler/100_vids/videos/"+vid
    return path


#%%
def correct_name(name):
    new= "".join(name.split("("))
    new="".join(new.split(")"))
    new="".join(new.split(" "))
    return new


#%%
#correct paths by removing paranthesis and spaces from filenames
for d,_,fs in os.walk(PATH):
    for f in fs:
        os.rename(os.path.join(d,f), correct_name(os.path.join(d,f)))


#%%
vids=[]
for d,_,fs in os.walk(PATH):
#     os.makedirs(change_to_new(d), exist_ok=True)
    for f in fs:
        if f.split(".")[-1]=="avi":
            vids.append(os.path.join(d,f))
            
len(vids)


#%%
len(np.unique(vids))


#%%
for vid in tqdm.tqdm(vids):
    command=f"ffmpeg -i {vid} -vcodec libx264 -acodec aac {change_to_new(vid)}"
    r=subprocess.call(command, shell=True)


