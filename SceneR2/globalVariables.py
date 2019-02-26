class gbVars:
    def __init__(self):
        self._path_to_data = '../data/'
        self._path_to_vids = "Daimler/100_vids/videos/"
        self._path_to_pickled_labels = 'Daimler/100_vids/pickled_labels'
        self._path_to_csv = "Daimler/100_vids/csv/"
        self._video_player = 'vlc'
    
    @property
    def path_to_data(self): return self._path_to_data
    @path_to_data.setter
    def path_to_data(self, val): self._path_to_data = val
    
    @property
    def path_to_vids(self): return self.path_to_data+self._path_to_vids
    @path_to_vids.setter
    def path_to_vids(self, val): self._path_to_vids = val

    @property
    def path_to_pickled_labels(self): return self.path_to_data+self._path_to_pickled_labels
    @path_to_pickled_labels.setter
    def path_to_pickled_labels(self, val): self._path_to_pickled_labels = val

    @property
    def path_to_csv(self): return self.path_to_data+self._path_to_csv
    @path_to_csv.setter
    def path_to_csv(self, val): self._path_to_csv = val
    
    @property
    def video_player(self): return self._video_player
    @video_player.setter
    def video_player(self, val): self._video_player = val
    
globalVariables=gbVars()