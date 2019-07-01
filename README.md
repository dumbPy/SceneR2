# SceneR2

A scene recognition, pedestrian detection and vehicle action detection model, with an extensive dataset encapsulation API. The API is written for proprietary data by *Daimler*.

# SceneR2WebApp

A web application in Django that takes a video and corresponding CAN data from Daimler's dataset, and processes them, shows the object detection processed video with [synced CAN](SceneR2WebApp/webapp/media/processed/can_slider.png) as a slider
![Webapp Demo](SceneR2WebApp/webapp/media/SceneR2WebApp.gif)
 
NOTE: This code is useless for anyone except the sponsors of the project who have the coresponding data.

### To Run

* If you don't have anaconda already, install miniconda from [here](https://repo.continuum.io/miniconda/)  
* Clone the repo with `git clone https://github.com/dumbPy/SceneR2.git`  
* Change directory to repo with `cd SceneR2`  
* clone the anaconda environment with `conda env update`. This would take some time to download all the packages inside a new conda environment `scener2-env`.  
* Once new package environment is made, activate it with `conda activate scener2-env`  
* Now you can run the web app with  
```
$ cd SceneR2WebApp  
$ python manage.py runserver 0.0.0.0:8000  
```
 This would run the SceneR2 Web Application server on your machine that can be either accesesed from same machine's browser at `localhost:8000` or from any other machine on the network with `<your-local-ip>:8000`


### Credits

Under the Supervison of **Prof. Manjesh Hanawal**<sup>1</sup>,  **Prof. P Balamurugan**<sup>1</sup> and **Dr. Tilak Singh**<sup>2</sup>\
<sup>**1**</sup>Indian Institute of Technology Bombay &nbsp;   &nbsp;  <sup>**2**</sup>Mitsubishi Fuso Truck and Bus Corp. Japan

